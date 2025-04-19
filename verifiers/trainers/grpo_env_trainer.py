import datetime
import json
import os
import time
from typing import Any, Callable, Optional, Union

import torch
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset

# from bespokelabs.curator.client import Client
from huanzhi_utils import load_file
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

from verifiers.envs.bfcl_inthinking_env import BfclITEnv
from verifiers.envs.environment import Environment
from verifiers.tools.bfcl_tools import INVOLVED_CLASS_TO_FUNC_DOC_PATH
from verifiers.utils.logging_utils import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig  # type: ignore

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        env: Environment,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        debug_generate: bool = False,
        debug_rewards: bool = False,
        run_name: str = "",
        model_name: str = "",
        use_dr_grpo: bool = False,
        test_hypothesis_clip_advantage: bool = False,
        apply_overlong_filtering: bool = False,
        print_sample_completions: bool = True,
        **kwargs,
    ):
        if not args.use_vllm:  # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (
            callable(reward_funcs)
            or (
                isinstance(reward_funcs, list)
                and all(callable(f) for f in reward_funcs)
            )
        ):
            raise ValueError(
                "reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models."
            )
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env
        self.debug_generate = debug_generate
        self.debug_rewards = debug_rewards
        self._eval_started = False
        self._train_started = False
        self.train_prompt_to_log = []
        self.train_completion_to_log = []
        self.train_reward_to_log = []
        self.train_dataset_rows_to_log = []
        self.eval_prompt_to_log = []
        self.eval_completion_to_log = []
        self.eval_reward_to_log = []
        self.eval_dataset_rows_to_log = []
        self.model_name = model_name
        self._initial_eval = True
        self.run_name = run_name
        self.use_dr_grpo = use_dr_grpo
        self.test_hypothesis_clip_advantage = test_hypothesis_clip_advantage
        self.apply_overlong_filtering = apply_overlong_filtering
        self.print_sample_completions = print_sample_completions

        if train_dataset is not None:
            dataset_hash = train_dataset._fingerprint
        else:
            dataset_hash = "N/A"
        metadata_dict = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "prompt_func": "N/A",
            "parse_func": "N/A",
            "model_name": model_name,
            "run_hash": run_name,
            "batch_mode": False,
            "response_format": "N/A",
        }
        # if os.environ["CURATOR_VIEWER"] == "1":
        #     self._curator_viewer_client = Client()
        #     self._curator_session_id = self._curator_viewer_client.create_session(metadata_dict)

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # NOTE: Each row in inputs is each row of the dataset
        prompts = [x["prompt"] for x in inputs]  # type: ignore
        # Construct the prompt to be displayed in the console
        prompts_to_display = []
        for x in inputs:
            prompt_to_display = f"First User Request: {x['prompt'][-1]['content']}\n\n"
            prompt_to_display += (
                f"User Question Bank: {json.loads(x['user_question_bank'])}\n"
                if len(json.loads(x["user_question_bank"])) > 0
                else ""
            )
            prompt_to_display += f"Involved Classes: {x['involved_classes']}\n\n"
            prompt_to_display += "Function Documentation:\n\n"
            for class_name in x["involved_classes"]:
                tools = []
                func_doc = load_file(INVOLVED_CLASS_TO_FUNC_DOC_PATH[class_name])
                for func in func_doc:
                    tools.append(func["name"])
                prompt_to_display += f"Tools for {class_name}: {tools}\n\n"
            for i, answer in enumerate(json.loads(x["answer"])):
                prompt_to_display += f"Ground Truth for Turn {i}: {answer}\n"
            prompts_to_display.append(prompt_to_display)
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]  # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,  # type: ignore
        )  # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)  # type: ignore
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        all_prompts = gather_object(prompts)
        all_inputs = gather_object(inputs)

        if self.accelerator.is_main_process:
            if isinstance(self.env, BfclITEnv):
                env_result = self.env.generate(
                    prompts=all_prompts,
                    llm=self.llm,
                    sampling_params=self.sampling_params,
                    dataset_rows=all_inputs,
                    debug=self.debug_generate,
                )
            else:
                env_result = self.env.generate(
                    prompts=all_prompts,
                    llm=self.llm,
                    sampling_params=self.sampling_params,
                )
            completion_ids = env_result["ids"]
            completion_messages = env_result["messages"]
            completion_mask = env_result["mask"]
            states = env_result["states"]
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)
            states = [None] * len(all_prompts)
        # raise ValueError("Stop Before Computing Rewards")

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        states = broadcast_object_list(states, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        states = states[process_slice]

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id
        )  # type: ignore

        completion_mask = [
            torch.tensor(mask, device=device) for mask in completion_mask
        ]
        completion_mask = pad(completion_mask, padding_value=0)

        # Applying overlong filtering by detecting stop strings
        if self.apply_overlong_filtering:
            # Define the stop strings to check for
            stop_strings = ["</tool>", "<TASK_FINISHED>", "<TASK_ERROR>"]

            completion_mask_filtered = []
            for i, (completion_id, mask) in enumerate(
                zip(completion_ids, completion_mask)
            ):
                # Find the non-pad token indices
                non_pad_indices = (
                    (completion_id != self.processing_class.pad_token_id)
                    .nonzero()
                    .squeeze()
                )

                # If there are no non-pad tokens, keep the mask as is
                if len(non_pad_indices) == 0:
                    completion_mask_filtered.append(mask)
                    continue

                # Get the last 10 non-pad tokens (or all if less than 10)
                last_n = min(10, len(non_pad_indices))
                last_tokens = completion_id[non_pad_indices[-last_n:]]

                # Decode the last tokens to check for stop strings
                last_tokens_str = self.processing_class.decode(
                    last_tokens, skip_special_tokens=False
                )

                # Check if any stop string is in the decoded text
                valid_ending = any(
                    stop_str in last_tokens_str for stop_str in stop_strings
                )

                if valid_ending:
                    completion_mask_filtered.append(mask)
                else:
                    # If the completion doesn't end with a valid stop string, mask it out
                    new_mask = torch.zeros_like(mask)
                    completion_mask_filtered.append(new_mask)

                    # Log the filtered completion
                    os.makedirs(f"outputs/eval_results/{self.run_name}", exist_ok=True)
                    with open(
                        f"outputs/eval_results/{self.run_name}/overlong_filtered_completions_by_stop_strings.txt",
                        "a",
                    ) as f:
                        f.write(
                            f"Completion ID: {completion_id}\n"
                            f"Last tokens: {last_tokens}\n"
                            f"Last tokens decoded: {last_tokens_str}\n"
                            f"Completion: {completion_messages[i]}\n"
                            f"Mask: {mask}\n"
                            f"New Mask: {new_mask}\n"
                            f"---\n"
                        )

            # Replace the original completion_mask with the filtered version
            completion_mask_filtered = torch.stack(completion_mask_filtered)
            assert completion_mask_filtered.shape == completion_mask.shape, (
                f"Completion Mask Shape: {completion_mask.shape}, Filtered Mask Shape: {completion_mask_filtered.shape}"
            )
            completion_mask = completion_mask_filtered.clone()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )
        # use message dicts for reward function inputs
        completions = completion_messages
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]  # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}  # type: ignore
            if isinstance(self.env, BfclITEnv):
                output_reward_func = reward_func(
                    completions=completions, states=states, debug=self.debug_rewards
                )
            else:
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )  # type: ignore
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        rewards_per_func = gather(rewards_per_func)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(
            dim=1
        )

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)  # type: ignore
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)  # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )  # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )  # type: ignore
        if self.use_dr_grpo:
            advantages = rewards - mean_grouped_rewards  # NOTE: Dr.GRPO Adjustment
        else:
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        if self.test_hypothesis_clip_advantage:
            advantages = torch.clip(
                advantages, min=0
            )  # Clip the advantages to be all positive
            # advantages = torch.clip(advantages, max=0) # Clip the advantages to be all negative
            assert (advantages >= 0).all(), f"Advantages: {advantages}"
            # assert (advantages <= 0).all(), f"Advantages: {advantages}"

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        # Calculate correctness based on unified_reward_func
        # Find the index of unified_reward_func
        unified_idx = None
        for i, reward_func in enumerate(self.reward_funcs):
            if reward_func.__name__ == "unified_reward_func":  # type: ignore
                unified_idx = i
                break

        # Calculate correctness values for each sample if unified_reward_func exists
        if unified_idx is not None:
            correctness = (rewards_per_func[:, unified_idx] >= 1.0).float()

        curr_batch_accuracy = (
            sum(correctness.cpu().tolist()) / len(correctness.cpu().tolist())
            if len(correctness.cpu().tolist()) > 0
            else 0
        )
        if "batch_accuracy" not in self._metrics[mode]:
            self._metrics[mode]["batch_accuracy"] = []
        self._metrics[mode]["batch_accuracy"].append(curr_batch_accuracy)

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )  # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        # NOTE: Already Gathered Rewards
        reward_per_func_mean = rewards_per_func.mean(0)  # type: ignore
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__  # type: ignore
            self._metrics[mode][f"rewards/{reward_func_name}"].append(
                reward_per_func_mean[i].item()
            )

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_to_display)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()

            # Gather individual reward components for logging
            rewards_per_func_to_log = rewards_per_func.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    if self.print_sample_completions:
                        print_prompt_completions_sample(
                            [prompts_to_log[0]],
                            [completions_to_log[0]],
                            [rewards_to_log[0]],
                            self.state.global_step,
                        )
                # Check if global step has changed since last logging
                if (
                    not hasattr(self, "last_logged_step")
                    or self.state.global_step != self.last_logged_step
                ):
                    # Save any accumulated data
                    if self.train_prompt_to_log:
                        self.save_mode_logs(
                            "train",
                            self.train_prompt_to_log,
                            self.train_completion_to_log,
                            self.train_reward_to_log,
                            self.train_dataset_rows_to_log,
                        )
                        self.train_prompt_to_log = []
                        self.train_completion_to_log = []
                        self.train_reward_to_log = []
                        self.train_dataset_rows_to_log = []
                    if self.eval_prompt_to_log:
                        self.save_mode_logs(
                            "eval",
                            self.eval_prompt_to_log,
                            self.eval_completion_to_log,
                            self.eval_reward_to_log,
                            self.eval_dataset_rows_to_log,
                        )
                        self.eval_prompt_to_log = []
                        self.eval_completion_to_log = []
                        self.eval_reward_to_log = []
                        self.eval_dataset_rows_to_log = []

                    # Update last logged step
                    self.last_logged_step = self.state.global_step
                    print(f"Cleared and saved logs at step {self.state.global_step}")

                # Always collect current data
                if mode == "train":
                    self.train_prompt_to_log.extend(prompts_to_log)
                    self.train_completion_to_log.extend(completions_to_log)
                    self.train_reward_to_log.extend(rewards_to_log)
                    self.train_dataset_rows_to_log.extend(all_inputs)
                    print(
                        f"Stored {len(prompts_to_log)} train samples (total: {len(self.train_prompt_to_log)})"
                    )
                elif mode == "eval":
                    self.eval_prompt_to_log.extend(prompts_to_log)
                    self.eval_completion_to_log.extend(completions_to_log)
                    self.eval_reward_to_log.extend(rewards_to_log)
                    self.eval_dataset_rows_to_log.extend(all_inputs)
                    print(
                        f"Stored {len(prompts_to_log)} eval samples (total: {len(self.eval_prompt_to_log)})"
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):  # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    # Add individual reward components to the table
                    for i, reward_func in enumerate(self.reward_funcs):
                        reward_func_name = reward_func.__name__  # type: ignore
                        table[f"{reward_func_name}"] = [
                            row[i] for row in rewards_per_func_to_log
                        ]

                    # Add correctness to the table if unified_reward_func exists
                    if unified_idx is not None:
                        table["correctness"] = [
                            (row[unified_idx] >= 1.0) for row in rewards_per_func_to_log
                        ]
                    table["contains_gibberish"] = [
                        (row[unified_idx] == -1) for row in rewards_per_func_to_log
                    ]
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})  # type: ignore

        # Clean up any existing instances before new generation
        if self.env.env_instances != {}:
            if self.debug_rewards or self.debug_generate:
                print(f"Found existing instances: {self.env.env_instances}")
                print("Cleaning up any existing instances before new generation")
                time.sleep(3)
            self.env.cleanup_instances()
            self.env.env_instances = {}

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def save_mode_logs(self, mode, prompts, completions, rewards, dataset_rows):
        """Helper method to save logs for a specific mode."""
        import pandas as pd

        print(f"Saving {mode} results with {len(prompts)} samples.")

        df = pd.DataFrame(
            {
                "step": [self.state.global_step] * len(prompts),
                "prompt": prompts,
                "completion": completions,
                "reward": rewards,
                "correctness": [r >= 1.0 for r in rewards],
                "contains_gibberish": [r == -1 for r in rewards],
                "dataset_rows": dataset_rows,
                "train": [mode == "train"] * len(prompts),
            }
        )

        current_time = datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=-8))
        ).strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.model_name.split("/")[-1].replace("-", "_")
        output_dir = f"outputs/eval_results/{self.run_name}/{current_time}_{mode}_{self.state.global_step}"

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save files with timestamp in the path
        df.to_csv(
            f"{output_dir}/{mode}_result_{model_name_safe}_step_{self.state.global_step}.csv",
            index=False,
        )
        dataset = Dataset.from_pandas(df)
        dataset.save_to_disk(
            f"{output_dir}/{mode}_result_{model_name_safe}_step_{self.state.global_step}.hf"
        )

        if self.state.global_step == 0 and mode == "eval":
            self._initial_eval = False

        # if os.environ["CURATOR_VIEWER"] == "1":
        #     from bespokelabs.curator.utils import push_to_viewer

        #     df.drop(columns=["dataset_rows"], inplace=True)
        #     dataset = Dataset.from_pandas(df)
        #     push_to_viewer(dataset, session_id=self._curator_session_id)

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        if self.use_dr_grpo:
            # NOTE: Dr.GRPO Adjustment
            loss = (
                torch.mean(torch.sum(per_token_loss * completion_mask, dim=-1))
                / self.args.max_completion_length
            )
        else:
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).nanmean().item()
            )

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).nanmean().item()
        )
        return loss
