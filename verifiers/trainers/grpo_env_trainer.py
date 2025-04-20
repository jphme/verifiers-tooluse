import datetime
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union # Added Dict, List, Sequence

import torch
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset

# from bespokelabs.curator.client import Client
from huanzhi_utils import load_file
from loguru import logger # Use logger
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

# Make sure imports point to the *modified* environment
from verifiers.envs.bfcl_inthinking_env import BfclITEnv
from verifiers.envs.environment import Environment
from verifiers.tools.bfcl_tools import INVOLVED_CLASS_TO_FUNC_DOC_PATH
from verifiers.utils.logging_utils import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig  # type: ignore

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def fix_r1_chat_template(chat_template: str) -> str:
    #the default chat template removes the think part of the think-step-response chain
    # we need this to inject into chain-of-thought
    problematic_part = """{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}"""
    # --- Define the corrected part ---
    corrected_part = """{% set content = message['content'] %}{{'<｜Assistant｜>' + content + ''}}"""
    modified_template = chat_template.replace(problematic_part, corrected_part)
    assert modified_template != chat_template, "Chat template not modified"
    return modified_template


class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        env: Environment, # Should be BfclITEnv instance now
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
        if not isinstance(env, BfclITEnv):
             logger.warning(f"Expected env to be BfclITEnv, but got {type(env)}. Ensure it follows the single-turn protocol.")

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
        # Ensure the env also has the tokenizer
        if (isinstance(model,str) and "R1" in model) or "R1" in model.name_or_path:
            self.tokenizer.chat_template = fix_r1_chat_template(self.tokenizer.chat_template)
            self.llm.set_tokenizer(self.tokenizer)
            logger.info("Chat Template fixed for R1 style models")
        if hasattr(self.env, 'tokenizer') and self.env.tokenizer is None:
            self.env.tokenizer = self.tokenizer
        elif not hasattr(self.env, 'tokenizer'):
             logger.warning("Environment does not have a 'tokenizer' attribute. Injecting.")
             self.env.tokenizer = self.tokenizer


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

        if train_dataset is not None and hasattr(train_dataset, '_fingerprint'):
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
            "batch_mode": False, # Keep as is?
            "response_format": "single_assistant_turn_with_tools", # Describe new format
        }
        # Set sampling params based on env defaults if not overridden
        # These should match the env's sampling_args
        self.sampling_params.stop = self.env.sampling_args.get("stop", ["</tool>", "<TASK_FINISHED>", "<TASK_ERROR>", "<|im_end|>"])
        self.sampling_params.include_stop_str_in_output = self.env.sampling_args.get("include_stop_str_in_output", True)
        self.sampling_params.ignore_eos = self.env.sampling_args.get("ignore_eos", False)
        logger.info(f"Trainer sampling params set: stop={self.sampling_params.stop}, include_stop={self.sampling_params.include_stop_str_in_output}, ignore_eos={self.sampling_params.ignore_eos}")

        # Curator viewer setup (optional)
        # if os.environ.get("CURATOR_VIEWER") == "1":
        #     try:
        #         from bespokelabs.curator.client import Client
        #         self._curator_viewer_client = Client()
        #         self._curator_session_id = self._curator_viewer_client.create_session(metadata_dict)
        #         logger.info(f"Curator viewer session created: {self._curator_session_id}")
        #     except ImportError:
        #         logger.warning("Curator client not found, viewer disabled.")
        #         self._curator_viewer_client = None
        # else:
        #     self._curator_viewer_client = None


    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]] # inputs are dicts from dataset rows
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # Extract prompts (list of message dicts) and dataset rows
        prompts = [x["prompt"] for x in inputs] # List[List[Dict[str, str]]]
        dataset_rows = list(inputs) # List[Dict[str, Any]]

        # Construct the prompt to be displayed in the console (more detailed)
        prompts_to_display = []
        for x in dataset_rows:
            prompt_display = f"--- Task ID: {x.get('id', 'N/A')} ---\n"
            # Display the user message content (which now includes instructions)
            user_content = x['prompt'][0]['content']
            prompt_display += f"User Request (contains instructions & tools):\n{user_content[:500]}...\n{user_content[-200:]}\n\n" # Show start and end
            prompt_display += f"Involved Classes: {x.get('involved_classes', 'N/A')}\n"
            # Display ground truth for this specific turn/request
            try:
                 gt_answer = json.loads(x.get("answer", "[]"))
                 prompt_display += f"Ground Truth Action(s) for this turn: {gt_answer}\n"
            except json.JSONDecodeError:
                 prompt_display += f"Ground Truth Action(s) (raw): {x.get('answer', 'N/A')}\n"
            prompts_to_display.append(prompt_display)

        # Apply chat template to get flat prompt text for tokenization
        # Ensure the tokenizer has the correct chat template set
        prompts_text = []
        prompts_text = []
        for example_messages in prompts: # example_messages is List[Dict[str, str]]
             try:
                 # Directly apply the template to the list of messages
                 # We need the string output for tokenization later
                 # add_generation_prompt=False because the llm.chat call handles the specific
                 # assistant prompt token when needed. Applying it here might cause issues.
                 templated_prompt = self.processing_class.apply_chat_template(
                     example_messages,
                     tokenize=False,
                     add_generation_prompt=False
                 )
                 prompts_text.append(templated_prompt)
             except Exception as e:
                  logger.exception(f"Error applying chat template directly to: {example_messages}. Error: {e}")
                  # Fallback or re-raise - using empty string might hide issues
                  prompts_text.append("") # Add empty string on error to maintain list length


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

        # Apply max_prompt_length constraint *after* tokenization if needed (redundant if truncation=True)
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Sync model if needed (vLLM specific)
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather prompts (original message dict format) and full dataset rows across processes
        all_prompts = gather_object(prompts)
        all_dataset_rows = gather_object(dataset_rows) # Gather the full rows

        # --- Generate Completions using the Environment ---
        if self.accelerator.is_main_process:
            logger.info(f"Main process generating completions for {len(all_prompts)} prompts...")
            # Pass the full dataset rows to generate
            env_result = self.env.generate(
                prompts=all_prompts,
                llm=self.llm,
                sampling_params=self.sampling_params,
                dataset_rows=all_dataset_rows, # Pass the gathered dataset rows
                debug=self.debug_generate,
            )
            # Extract results - check structure matches env output
            completion_ids_list = env_result["ids"] # List[List[int]]
            completion_messages_list = env_result["messages"] # List[List[Dict[str,str]]] (inner list has 1 assistant msg)
            completion_mask_list = env_result["mask"] # List[List[int]]
            states_list = env_result["states"] # List[Dict]
            logger.info(f"Main process finished generation. Received {len(completion_ids_list)} results.")
        else:
            # Placeholders for non-main processes
            completion_ids_list = [None] * len(all_prompts)
            completion_messages_list = [None] * len(all_prompts)
            completion_mask_list = [None] * len(all_prompts)
            states_list = [None] * len(all_prompts)

        # Broadcast results from main process to all processes
        logger.info(f"Process {self.accelerator.process_index}: Broadcasting results...")
        completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)
        completion_messages_list = broadcast_object_list(completion_messages_list, from_process=0)
        completion_mask_list = broadcast_object_list(completion_mask_list, from_process=0)
        states_list = broadcast_object_list(states_list, from_process=0)
        logger.info(f"Process {self.accelerator.process_index}: Broadcast finished.")


        # --- Process Broadcasted Results ---
        # Get the slice of results relevant to the current process
        process_batch_size = len(prompts) # Size of the batch this process handled initially
        process_slice = slice(
            self.accelerator.process_index * process_batch_size,
            (self.accelerator.process_index + 1) * process_batch_size,
        )

        local_completion_ids = completion_ids_list[process_slice]
        local_completion_messages = completion_messages_list[process_slice] # List[List[Dict]]
        local_completion_mask = completion_mask_list[process_slice]
        local_states = states_list[process_slice] # List[Dict]

        # Extract the actual completion text (single assistant message content)
        # Handle potential errors where message structure is wrong
        local_completions_text = []
        for msg_list in local_completion_messages:
             if msg_list and isinstance(msg_list, list) and len(msg_list) > 0 and msg_list[0].get('role') == 'assistant':
                 local_completions_text.append(msg_list[0]['content'])
             else:
                 logger.error(f"Unexpected completion message structure: {msg_list}. Using empty string.")
                 local_completions_text.append("") # Fallback for reward func


        # Pad completion IDs and masks for tensor operations
        completion_ids_tensors = [torch.tensor(ids, device=device, dtype=torch.long) for ids in local_completion_ids]
        completion_ids_padded = pad(
            completion_ids_tensors, padding_value=self.processing_class.pad_token_id
        )

        completion_mask_tensors = [torch.tensor(mask, device=device, dtype=torch.long) for mask in local_completion_mask]
        completion_mask_padded = pad(completion_mask_tensors, padding_value=0) # Pad mask with 0

        # --- Overlong Filtering (Optional) ---
        if self.apply_overlong_filtering:
            # This logic might need adjustment for the new format.
            # Checking for stop strings *within* the final assistant message.
            stop_strings = self.sampling_params.stop
            filtered_mask_list = []
            num_filtered = 0
            for i, completion_text in enumerate(local_completions_text):
                mask_tensor = completion_mask_padded[i]
                # Check if the *very end* of the text contains a valid stop token
                # (The env should ideally ensure this, but double check)
                valid_ending = any(completion_text.rstrip().endswith(stop) for stop in stop_strings)
                if valid_ending:
                    filtered_mask_list.append(mask_tensor)
                else:
                    # If no valid stop token at the end, consider it overlong/incomplete
                    logger.warning(f"Filtering completion (no valid stop token at end): ...{completion_text[-100:]}")
                    num_filtered += 1
                    # Mask out the entire completion by returning zeros
                    filtered_mask_list.append(torch.zeros_like(mask_tensor))
                    # Log filtered completion? (Be careful about log volume)

            if num_filtered > 0:
                 logger.info(f"Applied overlong filtering: Masked out {num_filtered}/{len(local_completions_text)} completions.")
                 completion_mask_padded = torch.stack(filtered_mask_list)
            # Ensure shape consistency
            assert completion_mask_padded.shape == completion_ids_padded.shape, \
                 f"Shape mismatch after filtering: Mask {completion_mask_padded.shape}, IDs {completion_ids_padded.shape}"


        # --- Prepare Tensors for Log Prob Calculation ---
        # Concatenate prompt and completion IDs/masks
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids_padded], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask_padded], dim=1)

        logits_to_keep = completion_ids_padded.size(1) # Only need logps for completion part

        # --- Calculate Log Probabilities ---
        with torch.no_grad():
            # Old logps (from policy model before update)
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                # If num_iterations is 1, old logps are same as current ones (calculated later)
                old_per_token_logps = None # Will use detached current logps later

            # Reference logps (from ref model or disabled adapter)
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else: # Use base model without adapter
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # --- Calculate Rewards ---
        # Use the single completion text and final states
        completions_for_reward = local_completions_text # List[str]
        states_for_reward = local_states # List[Dict]

        # Gather rewards across all processes *after* local calculation
        all_rewards_per_func_local = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )

        for i, reward_func in enumerate(self.reward_funcs):
            try:
                # Pass completions (text) and states (dicts)
                # Reward functions must be adapted to handle this format
                output_reward_func = reward_func(
                    completions=completions_for_reward,
                    states=states_for_reward,
                    debug=self.debug_rewards
                )
                all_rewards_per_func_local[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )
            except Exception as e:
                 logger.exception(f"Error calling reward function {reward_func.__name__}: {e}")
                 # Handle error: assign default reward (e.g., 0 or min value) or raise
                 all_rewards_per_func_local[:, i] = 0.0 # Assign 0 on error


        # Gather rewards from all processes
        gathered_rewards_per_func = gather(all_rewards_per_func_local) # Shape: (total_batch_size, num_reward_funcs)

        # Apply weights and sum to get final reward per sample
        # Ensure reward_weights is on the correct device
        reward_weights_tensor = self.reward_weights.to(gathered_rewards_per_func.device)
        gathered_rewards = (gathered_rewards_per_func * reward_weights_tensor.unsqueeze(0)).sum(dim=1)
        # Shape: (total_batch_size,)

        # --- Calculate Advantages ---
        # Compute grouped-wise rewards (assuming num_generations=1 here, otherwise needs adjustment)
        # If num_generations > 1, view/reshape based on that. Assuming 1 for now.
        if self.num_generations > 1:
             # Reshape assumes data is ordered [prompt1_gen1, prompt1_gen2, ..., promptN_genM]
             gathered_rewards_grouped = gathered_rewards.view(-1, self.num_generations)
             mean_grouped_rewards = gathered_rewards_grouped.mean(dim=1)
             std_grouped_rewards = gathered_rewards_grouped.std(dim=1)
             # Repeat means/stds to match original shape
             mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
             std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        else:
             # If only 1 generation per prompt, mean is the reward itself, std is 0
             mean_grouped_rewards = gathered_rewards
             std_grouped_rewards = torch.zeros_like(gathered_rewards)


        # Calculate advantages
        if self.use_dr_grpo:
            advantages = gathered_rewards - mean_grouped_rewards # Dr.GRPO: R - mean(R)
        else:
            # Standard GRPO: (R - mean(R)) / (std(R) + eps)
            advantages = (gathered_rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-5) # Increased epsilon slightly

        # Clip advantages (optional, for hypothesis testing)
        if self.test_hypothesis_clip_advantage:
            advantages = torch.clamp(advantages, min=0) # Example: clip to positive
            # assert (advantages >= 0).all(), f"Advantages should be non-negative: {advantages}"

        # Slice advantages to keep only the part for the current process
        local_advantages = advantages[process_slice]

        # --- Logging Metrics ---
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["reward"].append(gathered_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item()) # Log mean of stds

        # Log individual reward components
        reward_per_func_mean = gathered_rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = getattr(reward_func, '__name__', f'reward_func_{i}')
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func_mean[i].item())

        # Calculate and log accuracy based on a specific reward function (e.g., 'unified_reward_func')
        unified_idx = None
        for i, reward_func in enumerate(self.reward_funcs):
            if getattr(reward_func, '__name__', '') == "unified_reward_func":
                unified_idx = i
                break

        if unified_idx is not None:
            # Use the gathered rewards per function
            correctness = (gathered_rewards_per_func[:, unified_idx] >= 1.0).float()
            curr_batch_accuracy = correctness.mean().item() # Mean over the total batch
            if "batch_accuracy" not in self._metrics[mode]: self._metrics[mode]["batch_accuracy"] = []
            self._metrics[mode]["batch_accuracy"].append(curr_batch_accuracy)

        # Log completion length (use local mask before gathering)
        local_completion_length = completion_mask_padded.sum(1).float().mean().item()
        gathered_completion_length = self.accelerator.gather(torch.tensor(local_completion_length, device=device)).mean().item()
        self._metrics[mode]["completion_length"].append(gathered_completion_length)


        # --- Log Completions and Rewards (on main process) ---
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            # Gather necessary data to main process for logging
            all_prompts_display = gather_object(prompts_to_display)
            all_completions_text = gather_object(local_completions_text)
            all_rewards_list = gathered_rewards.cpu().tolist()
            all_rewards_per_func_list = gathered_rewards_per_func.cpu().tolist()
            # all_dataset_rows already gathered

            if self.accelerator.is_main_process:
                if is_rich_available() and self.print_sample_completions:
                    # Print sample (e.g., first one)
                    print_prompt_completions_sample(
                        [all_prompts_display[0]],
                        [all_completions_text[0]],
                        [all_rewards_list[0]],
                        self.state.global_step,
                    )

                # Check if global step changed to trigger saving accumulated logs
                if not hasattr(self, "last_logged_step") or self.state.global_step != self.last_logged_step:
                    if self.train_prompt_to_log:
                        self.save_mode_logs("train", self.train_prompt_to_log, self.train_completion_to_log, self.train_reward_to_log, self.train_dataset_rows_to_log)
                        self.train_prompt_to_log, self.train_completion_to_log, self.train_reward_to_log, self.train_dataset_rows_to_log = [], [], [], []
                    if self.eval_prompt_to_log:
                        self.save_mode_logs("eval", self.eval_prompt_to_log, self.eval_completion_to_log, self.eval_reward_to_log, self.eval_dataset_rows_to_log)
                        self.eval_prompt_to_log, self.eval_completion_to_log, self.eval_reward_to_log, self.eval_dataset_rows_to_log = [], [], [], []
                    self.last_logged_step = self.state.global_step
                    logger.info(f"Saved accumulated logs at step {self.state.global_step}")

                # Accumulate current batch data
                target_prompt_log = self.train_prompt_to_log if mode == "train" else self.eval_prompt_to_log
                target_completion_log = self.train_completion_to_log if mode == "train" else self.eval_completion_to_log
                target_reward_log = self.train_reward_to_log if mode == "train" else self.eval_reward_to_log
                target_dataset_log = self.train_dataset_rows_to_log if mode == "train" else self.eval_dataset_rows_to_log

                target_prompt_log.extend(all_prompts_display)
                target_completion_log.extend(all_completions_text)
                target_reward_log.extend(all_rewards_list)
                target_dataset_log.extend(all_dataset_rows) # Log the full dataset row dict
                logger.info(f"Accumulated {len(all_prompts_display)} {mode} samples for logging (Total: {len(target_prompt_log)})")


                # Log to WandB if enabled
                if is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
                    import pandas as pd
                    table_data = {
                        "step": [str(self.state.global_step)] * len(all_rewards_list),
                        "prompt": all_prompts_display,
                        "completion": all_completions_text,
                        "reward": all_rewards_list,
                    }
                    # Add individual reward components
                    for i, reward_func in enumerate(self.reward_funcs):
                        reward_func_name = getattr(reward_func, '__name__', f'reward_func_{i}')
                        table_data[f"reward_{reward_func_name}"] = [row[i] for row in all_rewards_per_func_list]

                    # Add correctness/gibberish if unified_reward_func exists
                    if unified_idx is not None:
                        table_data["correctness"] = [(row[unified_idx] >= 1.0) for row in all_rewards_per_func_list]
                        table_data["contains_gibberish"] = [(row[unified_idx] == -1.0) for row in all_rewards_per_func_list] # Assuming -1 for gibberish

                    try:
                         df = pd.DataFrame(table_data)
                         wandb.log({"completions_table": wandb.Table(dataframe=df)}, step=self.state.global_step)
                    except Exception as e:
                         logger.error(f"Failed to log table to WandB: {e}")


        # --- Return Data for Loss Calculation ---
        # Return only the local portion relevant to this process
        return {
            "prompt_ids": prompt_ids, # Local prompt IDs
            "prompt_mask": prompt_mask, # Local prompt mask
            "completion_ids": completion_ids_padded, # Local completion IDs (padded)
            "completion_mask": completion_mask_padded, # Local completion mask (padded)
            "old_per_token_logps": old_per_token_logps, # Local old logps (or None)
            "ref_per_token_logps": ref_per_token_logps, # Local ref logps (or None)
            "advantages": local_advantages, # Local advantages
        }


    def save_mode_logs(self, mode, prompts, completions, rewards, dataset_rows):
        """Helper method to save logs for a specific mode. (Modified)"""
        if not prompts: # Don't save if nothing accumulated
             logger.info(f"No {mode} logs to save at step {self.state.global_step}.")
             return

        import pandas as pd
        logger.info(f"Saving {mode} results with {len(prompts)} samples at step {self.state.global_step}.")

        # Extract relevant info from dataset_rows if needed, or save the whole dict
        # Example: Extracting ID and ground truth
        ids = [row.get('id', f'unknown_{i}') for i, row in enumerate(dataset_rows)]
        ground_truth_answers = [row.get('answer', '[]') for row in dataset_rows]
        involved_classes = [row.get('involved_classes', []) for row in dataset_rows]

        # Calculate correctness/gibberish based on rewards (assuming unified_reward >= 1 is correct, -1 is gibberish)
        # This requires the reward logic to be consistent.
        correctness = [(r >= 1.0) if isinstance(r, (int, float)) else False for r in rewards]
        contains_gibberish = [(r == -1.0) if isinstance(r, (int, float)) else False for r in rewards]


        try:
            df_data = {
                "step": [self.state.global_step] * len(prompts),
                "id": ids,
                "prompt_display": prompts, # The formatted prompt string
                "completion": completions, # The generated assistant message content
                "reward": rewards,
                "correctness": correctness,
                "contains_gibberish": contains_gibberish,
                "ground_truth_answer": ground_truth_answers,
                "involved_classes": involved_classes,
                # Optionally include the full dataset row as JSON string if needed, but makes CSV large
                # "dataset_row_json": [json.dumps(row) for row in dataset_rows],
                "mode": [mode] * len(prompts),
            }
            df = pd.DataFrame(df_data)

            # Naming convention
            current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8))).strftime("%Y%m%d_%H%M%S") # PST example
            model_name_safe = self.model_name.split("/")[-1].replace("-", "_") if self.model_name else "unknown_model"
            output_dir = os.path.join(self.args.output_dir, f"eval_logs/{self.run_name}") # Save within main output dir

            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # File names
            base_filename = f"{mode}_results_{model_name_safe}_step_{self.state.global_step}_{current_time}"
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            hf_path = os.path.join(output_dir, f"{base_filename}.hf")

            # Save files
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {mode} results CSV to: {csv_path}")

            # Save as Hugging Face Dataset
            # Exclude potentially problematic columns for HF dataset conversion if needed
            # df_for_hf = df.drop(columns=["dataset_row_json"], errors='ignore') # Example
            dataset = Dataset.from_pandas(df)
            dataset.save_to_disk(hf_path)
            logger.info(f"Saved {mode} results HF Dataset to: {hf_path}")

        except Exception as e:
            logger.exception(f"Error saving {mode} logs: {e}")


        if self.state.global_step == 0 and mode == "eval":
            self._initial_eval = False

        # Curator push (optional)
        # if self._curator_viewer_client:
        #     try:
        #         from bespokelabs.curator.utils import push_to_viewer
        #         # Prepare dataset for curator (might need specific columns)
        #         curator_df = df[["step", "prompt_display", "completion", "reward", "correctness", "mode"]]
        #         curator_dataset = Dataset.from_pandas(curator_df)
        #         push_to_viewer(curator_dataset, session_id=self._curator_session_id)
        #         logger.info(f"Pushed {mode} results to Curator viewer.")
        #     except Exception as e:
        #         logger.error(f"Failed to push results to Curator: {e}")


    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None # Added num_items_in_batch
    ):
        """Computes GRPO loss using the generated data."""
        if return_outputs:
            raise ValueError("The GRPOEnvTrainer does not support returning outputs")

        # Unpack inputs from _generate_and_score_completions
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"] # These are padded
        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"]
        ref_per_token_logps = inputs["ref_per_token_logps"]

        # --- Compute Current Policy Log Probabilities ---
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1) # Only need logps for completion part

        # Calculate logps for the *current* policy model state
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        ) # Shape: (batch_size, completion_len)

        # --- Handle old_per_token_logps if num_iterations == 1 ---
        if self.num_iterations == 1:
            # In this case, old_logps weren't computed earlier.
            # Use the *detached* current logps as the 'old' logps for the ratio.
            old_per_token_logps = per_token_logps.detach()

        # --- Compute KL Divergence (if beta > 0) ---
        if self.beta != 0.0:
            if ref_per_token_logps is None:
                 raise ValueError("ref_per_token_logps is None but beta is non-zero. Ensure reference model is configured.")
            # KL divergence: exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
            # Ensure shapes match: per_token_logps vs ref_per_token_logps
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
        else:
            per_token_kl = 0.0 # No KL penalty if beta is 0

        # --- Compute GRPO Loss Components ---
        # Ratio r(theta) = exp(logp_policy - logp_old)
        log_ratio = per_token_logps - old_per_token_logps
        ratio = torch.exp(log_ratio)

        # Clipped ratio: clip(r(theta), 1 - epsilon, 1 + epsilon)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

        # Policy gradient loss terms (apply advantage)
        # Unsqueeze advantages to match shape (batch_size, 1) for broadcasting
        advantages_unsqueezed = advantages.unsqueeze(1)
        pg_loss1 = ratio * advantages_unsqueezed
        pg_loss2 = clipped_ratio * advantages_unsqueezed

        # Take the minimum of the two terms (conservative update)
        # Multiply by -1 because we maximize the objective (minimize negative objective)
        per_token_pg_loss = -torch.min(pg_loss1, pg_loss2)

        # Combine policy loss and KL penalty
        per_token_loss = per_token_pg_loss + self.beta * per_token_kl

        # --- Aggregate Loss ---
        # Apply completion mask to consider only valid tokens
        masked_per_token_loss = per_token_loss * completion_mask # Shape: (batch_size, completion_len)

        # Calculate loss per sequence
        loss_per_sequence = masked_per_token_loss.sum(dim=1) # Shape: (batch_size,)
        tokens_per_sequence = completion_mask.sum(dim=1) # Shape: (batch_size,)

        # Normalize loss
        if self.use_dr_grpo:
            # Dr.GRPO: Average loss per sequence, then average over batch
            # Normalize by max_completion_length? Or by actual tokens? Let's use actual tokens.
            # Avoid division by zero for sequences with no valid tokens
            safe_tokens_per_sequence = tokens_per_sequence.clamp(min=1.0)
            normalized_loss_per_sequence = loss_per_sequence / safe_tokens_per_sequence
            loss = normalized_loss_per_sequence.mean() # Mean over the batch
        else:
            # Standard GRPO: Sum loss over all tokens, divide by total number of valid tokens
            total_loss = masked_per_token_loss.sum()
            total_valid_tokens = completion_mask.sum().clamp(min=1.0)
            loss = total_loss / total_valid_tokens

        # --- Log Loss-Related Metrics ---
        mode = "eval" if self.control.should_evaluate else "train"

        # Log KL divergence if applicable
        if self.beta != 0.0:
            # Calculate mean KL per valid token
            masked_kl = per_token_kl * completion_mask
            mean_kl = masked_kl.sum() / completion_mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item() # Gather and average KL
            )

        # Log clipping ratio
        # is_clipped = (ratio < (1 - self.epsilon)) | (ratio > (1 + self.epsilon)) # Where clipping occurred
        is_clipped = (pg_loss1 > pg_loss2).float() # Where the clipped term was smaller (more accurate)
        masked_is_clipped = is_clipped * completion_mask
        clip_ratio = masked_is_clipped.sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather(clip_ratio).nanmean().item() # Gather and average clip ratio
        )

        # Log raw advantages mean/std
        self._metrics[mode]["advantages_mean"].append(
             self.accelerator.gather(advantages).nanmean().item()
        )
        self._metrics[mode]["advantages_std"].append(
             self.accelerator.gather(advantages).std().item()
        )

        # Log policy logp mean
        policy_logps_mean = (per_token_logps * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["policy_logps_mean"].append(
            self.accelerator.gather(policy_logps_mean).nanmean().item()
        )


        return loss