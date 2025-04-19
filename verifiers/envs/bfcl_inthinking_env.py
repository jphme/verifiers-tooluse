# bfcl_inthinking_env.py

import copy
import gc
import importlib
import inspect
import json
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch # Added for tokenization
from datasets import (
    Dataset,
    DatasetDict,  # type: ignore
)
from huanzhi_utils import load_file
from loguru import logger
from sklearn.model_selection import train_test_split
from trl.trainer.grpo_trainer import RewardFunc
from transformers import PreTrainedTokenizerBase # Added for tokenization

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.envs.tool_env import infer_schema_from_function
from verifiers.parsers import XMLParser
from verifiers.rubrics.bfcl_inthinking_rubric import BfclITRubric
from verifiers.tools.bfcl_tools import (
    INVOLVED_CLASS_TO_FUNC_DOC_PATH,
    construct_tools_from_involved_classes,
)

from ..imports import LLM, SamplingParams  # type: ignore

# New prompt format combining instructions and user query
BFCL_INTHINKING_USER_PROMPT = """You are an expert in composing functions. You are given a question from a user and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to complete the task.
You have access to the following tools to help solve the task:

{tools}

Follow these instructions precisely:
1. Think step-by-step which of the available functions need to be called in which order and with which parameters to fulfill the query and plan your actions.
2. If tool use is necessary, write one or more JSON commands as a list inside <tool> </tool> tags. Each item in the list must have "name" and "args" keys, with "args" being a dictionary.
   Example: <tool> [{{"name": "func_1_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}, {{"name": "func_2_name", "args": {{"arg3": "value3", "arg4": "value4"}}}}] </tool>
   Only use the provided tools and their specified arguments. Adhere strictly to the JSON format.
3. Tool results will be provided in <tool_result> </tool_result> tags.
4. After receiving tool results, continue your thought process if necessary. You can use tools again if needed, following step 2.
5. Once the task is fully addressed and no more tool calls are needed, stop your thinking with </think> and provide a final summary or answer. End your entire response with <TASK_FINISHED>.
6. If you encounter an unrecoverable error or determine the task cannot be completed with the available tools, stop your thinking with </think>, explain the issue and end your response with <TASK_ERROR>.

Here is the user question:
{user_query}"""


def format_bfcl_prompt(
    involved_classes: List[str] | None = None,
    user_question: str | None = None,
) -> List[Dict[str, str]]:
    messages = []
    tools = construct_tools_from_involved_classes(involved_classes)
    # Combine instructions, tools, and query into the first user message
    messages.append(
        {
            "role": "user",
            "content": BFCL_INTHINKING_USER_PROMPT.format(
                tools=tools, user_query=user_question
            ),
        }
    )
    return messages


def preprocess_bfcl_dataset(
    curriculum_learning: bool = False, split: str = "train"
) -> Dataset:
    # TODO: Change to local path
    multi_turn_base_data = load_file(
        "verifiers/berkeley-function-call-leaderboard/data/BFCL_v3_multi_turn_base.json"
    )
    multi_turn_base_answer = load_file(
        "verifiers/berkeley-function-call-leaderboard/data/possible_answer/BFCL_v3_multi_turn_base.json"
    )

    # Reprocess the columns into serializable format and add num_turns
    for i in range(len(multi_turn_base_data)):
        question_data = multi_turn_base_data[i]["question"]
        ground_truth = multi_turn_base_answer[i]["ground_truth"]
        initial_config = multi_turn_base_data[i]["initial_config"]

        # Assert number of turns matches between question and ground truth
        assert len(question_data) == len(ground_truth), (
            f"Mismatch in number of turns for entry {i}"
        )

        multi_turn_base_data[i]["num_turns"] = len(question_data)
        multi_turn_base_data[i]["question"] = json.dumps(question_data)
        multi_turn_base_data[i]["initial_config"] = json.dumps(initial_config)
        multi_turn_base_data[i]["answer"] = json.dumps(ground_truth)

    if curriculum_learning:
        # Create curriculum data with copies for each turn
        curriculum_data = []
        for entry in multi_turn_base_data:
            questions = json.loads(entry["question"])
            answers = json.loads(entry["answer"])

            # Create copies for each turn number
            for j in range(1, entry["num_turns"] + 1):
                curriculum_entry = copy.deepcopy(entry)
                curriculum_entry["question"] = json.dumps(copy.deepcopy(questions[:j]))
                curriculum_entry["answer"] = json.dumps(copy.deepcopy(answers[:j]))
                curriculum_entry["num_turns"] = j
                curriculum_data.append(curriculum_entry)
        multi_turn_base_data = curriculum_data

    dataset = Dataset.from_list(multi_turn_base_data)
    dataset = dataset.map(
        lambda x: {
            "prompt": format_bfcl_prompt(
                involved_classes=x["involved_classes"],
                user_question=json.dumps(json.loads(x["question"])[0][0]["content"]),
            ),
            # NOTE:: user_question_bank is a list of lists
            "user_question_bank": json.dumps(json.loads(x["question"])[1:])
            if len(json.loads(x["question"])) > 1
            else json.dumps([]),
            "ground_truth_bank": copy.deepcopy(x["answer"]),
            "num_turns": x["num_turns"],
            "id": x["id"],
        }
    )
    for i in range(len(dataset)):
        ground_truth_bank = json.loads(dataset[i]["ground_truth_bank"])
        user_question_bank = json.loads(dataset[i]["user_question_bank"])
        assert len(ground_truth_bank) == len(user_question_bank) + 1, (
            f"Length mismatch at index {i}: ground_truth_bank ({len(ground_truth_bank)}) != user_question_bank ({len(user_question_bank)})"
        )
    # Get unique IDs and split those first
    unique_ids = sorted(list(set(dataset["id"])))
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.5, random_state=42)

    # Filter dataset based on IDs
    train_dataset = dataset.filter(lambda x: x["id"] in train_ids)
    test_dataset = dataset.filter(lambda x: x["id"] in test_ids)

    if curriculum_learning:
        # Sort both splits by num_turns while preserving randomization within same num_turns
        def sort_by_turns(split):
            df = split.to_pandas()
            # Set seed for reproducibility
            rng = np.random.RandomState(42)
            # Randomize order within same num_turns by adding small random values
            df["sort_key"] = df["num_turns"] + rng.random(len(df)) * 0.1
            df = df.sort_values("sort_key")
            df = df.drop("sort_key", axis=1)
            return Dataset.from_pandas(df)

        train_dataset = sort_by_turns(train_dataset)
        test_dataset = sort_by_turns(test_dataset)

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    # assert train_dataset and test_dataset have non-overlapping ids
    assert len(set(train_dataset["id"]) & set(test_dataset["id"])) == 0, (
        "Train and test datasets have overlapping ids"
    )

    return dataset_dict[split]

class BfclITEnv(MultiStepEnv):
    def __init__(
        self,
        dataset: str = "bfcl",
        tools: List[Callable] = [],
        few_shot: List[Dict[str, str]] = [],
        sampling_args={ # Default stop tokens for the new flow
            "stop": [
                "</tool>",
                "<TASK_FINISHED>",
                "<TASK_ERROR>",
            ],
            "include_stop_str_in_output": True,
        },
        mask_env_response: bool = True, # Keep True to mask injected tool results
        max_steps_per_turn: int = 10, # Now max tool interactions per assistant turn
        curriculum_learning: bool = True,
        use_latest_trl: bool = False,
        **kwargs,
    ):
        logger.info("Initializing BfclITEnv (Single Assistant Turn Flow)")
        self.tokenizer = None # will be set later

        # max_num_turns is less relevant now for generation, but kept for dataset filtering
        self.max_num_turns = kwargs.pop("max_num_turns", 1)

        super().__init__(
            few_shot=few_shot,
            mask_env_response=mask_env_response, # Will mask injected tool results
            sampling_args=sampling_args,
            **kwargs,
        )
        self.CLASS_FILE_PATH_MAPPING = {
            "GorillaFileSystem": "verifiers.envs.bfcl_envs.gorilla_file_system",
            "MathAPI": "verifiers.envs.bfcl_envs.math_api",
            "MessageAPI": "verifiers.envs.bfcl_envs.message_api",
            "TwitterAPI": "verifiers.envs.bfcl_envs.posting_api",
            "TicketAPI": "verifiers.envs.bfcl_envs.ticket_api",
            "TradingBot": "verifiers.envs.bfcl_envs.trading_bot",
            "TravelAPI": "verifiers.envs.bfcl_envs.travel_booking",
            "VehicleControlAPI": "verifiers.envs.bfcl_envs.vehicle_control",
        }
        self.STATELESS_CLASSES = ["MathAPI"]
        self.env_instances = {}
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}

        self.dataset_name = dataset
        self.curriculum_learning = curriculum_learning

        self.max_steps_per_turn = max_steps_per_turn # Max tool interactions
        logger.info("Initializing Scoring Rubric")
        self.rubric = BfclITRubric()
        logger.info("Initializing LLM Parser")
        self.llm_parser = XMLParser(fields=["think", "tool"])
        # self.env_parser = XMLParser(fields=["tool_result"]) # No longer needed
        self.use_latest_trl = use_latest_trl
        # self.message_end_id = 151645 # Likely not needed anymore

    def get_dataset(self, max_num_turns: int = -1, **kwargs: Any) -> Dataset:
        if self.dataset is None:
            logger.info(f"Preprocessing dataset {self.dataset_name} for train split...")
            self.dataset = preprocess_bfcl_dataset(
                split="train",
                curriculum_learning=self.curriculum_learning,
            )
        # Filter based on original total turns in task if max_num_turns is provided
        effective_max_turns = max_num_turns if max_num_turns > 0 else self.max_num_turns
        if effective_max_turns > 0:
             logger.info(f"Filtering train dataset to max {effective_max_turns} total turns in task.")
             self.dataset = self.dataset.filter(
                 lambda x: x["num_total_turns_in_task"] <= effective_max_turns
             )
        return self.dataset

    def get_dataset(self, max_num_turns: int = -1, **kwargs: Any) -> Dataset:
        if self.dataset is None:
            self.dataset = preprocess_bfcl_dataset(
                split="train",
                curriculum_learning=self.curriculum_learning,
            )
        if max_num_turns > 0:
            self.dataset = self.dataset.filter(
                lambda x: x["num_turns"] <= max_num_turns
            )
        return self.dataset

    def get_eval_dataset(
        self,
        n: int = -1,
        max_num_turns: int = -1,
        max_turn_only: bool = False,
        **kwargs: Any,
    ) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_bfcl_dataset(
                split="test",
                curriculum_learning=self.curriculum_learning,
            )
        if max_num_turns > 0:
            self.eval_dataset = self.eval_dataset.filter(
                lambda x: x["num_turns"] <= max_num_turns
            )
        if max_turn_only:
            # Group by id and keep only max num_turns entry per group
            grouped = {}
            for item in self.eval_dataset:
                item_id = item["id"]
                if (
                    item_id not in grouped
                    or grouped[item_id]["num_turns"] < item["num_turns"]
                ):
                    grouped[item_id] = item
            self.eval_dataset = Dataset.from_list(list(grouped.values()))
        if n > 0:
            self.eval_dataset = self.eval_dataset.shuffle().select(range(n))
        return self.eval_dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def _get_tool_interaction_count(
        self, messages: List[Dict[str, str]], debug: bool = False
    ) -> int:
        """Counts the number of tool interactions within the assistant message."""
        count = 0
        if messages and messages[-1].get("role") == "assistant":
            count = messages[-1]["content"].count("</tool>") # Count occurrences of tool stop token
        if debug:
            print(f"Tool Interaction Count: {count}")
        return count

    def is_completed(
        self, state: Dict[str, Any] = None, debug: bool = False, **kwargs: Any
    ) -> bool:
        """Checks if the current assistant turn is complete."""
        # Check if already marked completed by step logic
        if state.get("completed", False):
             return True

        messages = state["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            # Not completed if no assistant message yet or last message is not assistant
            return False

        assistant_response = messages[-1]["content"]
        tool_interactions = self._get_tool_interaction_count(messages, debug=debug)

        # Check for termination tokens
        if "<TASK_FINISHED>" in assistant_response or "<TASK_ERROR>" in assistant_response:
            if debug:
                print(f"Found termination token. Entry Completed. Tool interactions: {tool_interactions}")
            return True

        # Check for max tool interactions
        if tool_interactions >= self.max_steps_per_turn:
            if debug:
                print(f"Reached max tool interactions ({self.max_steps_per_turn}). Entry Considered Completed.")
            # Optionally, force a TASK_ERROR or specific handling here
            # For now, just mark as completed to stop generation.
            return True

        # Check for max completion length (handled in step, but double-check)
        sampling_params = kwargs.get("sampling_params")
        if sampling_params and "max_tokens" in sampling_params:
             if len(state.get("completion_ids", [])) >= sampling_params.max_tokens:
                 if debug:
                     print(f"Reached max tokens ({sampling_params.max_tokens}). Entry Completed.")
                 return True

        # Otherwise, not completed
        return False

    # current_entry_completed is an alias for is_completed now
    current_entry_completed = is_completed

    # current_turn_completed is removed

    def call_tool(
        self,
        tool_json: str,
        state: Dict[str, Any] = None,
        debug: bool = False,
        ground_truth: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call a tool based on JSON command. (Modified for ground truth handling)"""
        if ground_truth:
            if debug:
                print("Executing Ground Truth Tool Call")
                # time.sleep(1) # Reduced sleep
            try:
                # Ground truth 'answer' is now a list containing one item (the list of calls for the turn)
                tool_calls_for_turn = json.loads(state["dataset_row"]["answer"])[0]
                if not isinstance(tool_calls_for_turn, list):
                    print(f"Error: Expected list of tool calls in ground truth, got: {tool_calls_for_turn}")
                    raise Exception("Error in ground truth tool execution format!")

                all_func_call_results = []
                method_to_instance = {}
                if "ground_truth_environment" not in state or not state["ground_truth_environment"]:
                     print("Error: Ground truth environment not initialized.")
                     raise Exception("Ground truth environment missing.")

                for class_name, instance in state["ground_truth_environment"].items():
                    for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
                        if not method_name.startswith("_"):
                            method_to_instance[method_name] = class_name

                # Process each function call *string* from the ground truth list
                for func_call_str in tool_calls_for_turn:
                    if not isinstance(func_call_str, str) or "(" not in func_call_str:
                        print(f"Error: Invalid ground truth function call format: {func_call_str}")
                        raise Exception("Error in ground truth tool execution format!")

                    method_name = func_call_str.split("(")[0].strip()
                    if method_name not in method_to_instance:
                        print(f"Error: Method '{method_name}' from ground truth not found in environment instances.")
                        print(f"Available methods: {list(method_to_instance.keys())}")
                        raise Exception("Error finding ground truth method!")

                    class_name = method_to_instance[method_name]
                    instance = state["ground_truth_environment"][class_name]
                    # IMPORTANT: Use the state dict directly in eval for safety
                    modified_call = f"state['ground_truth_environment']['{class_name}'].{func_call_str}"

                    if debug:
                        print(f"Executing ground truth call: {func_call_str}")
                        # time.sleep(1) # Reduced sleep
                    try:
                        # Use a restricted eval or preferably direct attribute access if possible
                        # For simplicity here, using eval but be cautious in production
                        result = eval(modified_call, {"state": state}) # Pass state explicitly
                        result_str = str(result) if result is not None else "Success"
                        all_func_call_results.append(
                            f"Function Call {func_call_str} Succeeded. Result: {result_str}"
                        )
                    except Exception as e:
                        print(f"Error executing ground truth call '{modified_call}': {e}")
                        # Decide how to handle errors - raise or return error message?
                        # For reward calculation, maybe best to raise to signal GT failure.
                        raise Exception(f"Error during ground truth tool execution: {e}")

                # No actual result string needed for GT, just update state
                return "Ground truth execution successful.", state
            except Exception as e:
                print(f"Ground Truth Execution Failed: {e}")
                # Log details for debugging
                print(f"State: {state.get('id', 'N/A')}")
                print(f"Ground Truth Answer: {state.get('dataset_row', {}).get('answer', 'N/A')}")
                raise Exception(f"Error in ground truth tool execution is not expected!! Error: {e}")


        # --- Handling Model Tool Calls (Remains largely the same) ---
        try:
            command = json.loads(tool_json)
            all_func_call_results = []
            if not isinstance(command, list):
                return json.dumps(["Error: Invalid tool command. Tool command must be one list of JSON objects."]), state
            if command == []:
                 return json.dumps(["Function Call Failed. Error: Found empty tool calls."]), state

            for tool_call in command:
                if not (
                    isinstance(tool_call, dict)
                    and "name" in tool_call
                    and "args" in tool_call
                    and isinstance(tool_call["args"], dict)
                ):
                    all_func_call_results.append(f"Function Call {tool_call} Failed. Error: Invalid format (must have 'name' and 'args' dict).")
                    # Stop processing further calls in this batch on format error
                    return json.dumps(all_func_call_results), state

                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Convert lists to tuples if needed by specific tools
                for key, value in tool_args.items():
                    if isinstance(value, list):
                        tool_args[key] = tuple(value)
                # tool_call["args"] = tool_args # No need to modify original dict here

                if debug:
                    print(f"Attempting Tool Call: {tool_name}({tool_args})")
                    # time.sleep(1)

                found_method = False
                target_instance = None
                if "environment" not in state or not state["environment"]:
                     return json.dumps([f"Function Call {tool_call} Failed. Error: Environment not initialized."]), state

                for class_instance in state["environment"].values():
                    if hasattr(class_instance, tool_name):
                        found_method = True
                        target_instance = class_instance
                        if debug:
                            print(f"Found method {tool_name} in class {target_instance.__class__.__name__}")
                        tool_func = getattr(target_instance, tool_name)
                        break

                if not found_method:
                    available_tools_msg = self._get_available_tools_message(state)
                    all_func_call_results.append(f"Function Call {tool_call} Failed. Error: Method '{tool_name}' not found. {available_tools_msg}")
                    # Stop processing further calls if a tool is not found
                    return json.dumps(all_func_call_results), state

                try:
                    result = tool_func(**tool_args)
                    result_str = str(result) if result is not None else "Success" # Handle None results

                    # Check for explicit errors returned by the tool itself
                    # Be careful with generic "error" checks
                    if isinstance(result, dict) and 'error' in result:
                         error_detail = result.get('error', 'Unknown tool error')
                         all_func_call_results.append(f"Function Call {tool_call} Failed during execution. Error: {error_detail}")
                         # Stop processing further calls if one fails
                         return json.dumps(all_func_call_results), state
                    elif isinstance(result, str) and "error" in result.lower() and len(result) < 100: # Heuristic for error strings
                         all_func_call_results.append(f"Function Call {tool_call} Failed during execution. Error: {result}")
                         # Stop processing further calls if one fails
                         return json.dumps(all_func_call_results), state
                    else:
                         all_func_call_results.append(f"Function Call {tool_call} Succeeded. Result: {result_str}")
                         # Log successful calls per interaction (might need adjustment)
                         if "successful_func_calls" not in state: state["successful_func_calls"] = []
                         # Append to the latest list if exists, else create new
                         if not state["successful_func_calls"] or not isinstance(state["successful_func_calls"][-1], list):
                             state["successful_func_calls"].append([tool_call])
                         else:
                             state["successful_func_calls"][-1].append(tool_call)

                except Exception as e:
                    import traceback
                    tb_str = traceback.format_exc()
                    error_msg = f"Function Call {tool_call} Failed during execution. Error: {e}. Traceback: {tb_str}"
                    logger.error(error_msg) # Log the full error
                    all_func_call_results.append(f"Function Call {tool_call} Failed during execution. Error: {e}")
                    # Stop processing further calls if one raises an exception
                    return json.dumps(all_func_call_results), state

            return json.dumps(all_func_call_results), state
        except json.JSONDecodeError as e:
            return json.dumps([f"Error decoding tool call JSON: {e}. Ensure the format is a list of JSON objects."]), state
        except Exception as e:
            logger.exception(f"Unexpected error in call_tool: {e}") # Log unexpected errors
            return json.dumps([f"Unexpected error processing tool call: {e}"]), state

    def _get_available_tools_message(self, state: Dict[str, Any]) -> str:
        """Helper to create a message listing available tools."""
        available_tools = []
        try:
            for class_name in state.get("dataset_row", {}).get("involved_classes", []):
                func_doc_path = INVOLVED_CLASS_TO_FUNC_DOC_PATH.get(class_name)
                if func_doc_path:
                    func_doc = load_file(func_doc_path)
                    for func in func_doc:
                        available_tools.append(func["name"])
            return f"Available Tools: {sorted(list(set(available_tools)))}" if available_tools else "No tools seem available."
        except Exception as e:
            logger.warning(f"Could not retrieve available tools: {e}")
            return "Could not determine available tools."

    # Needs to be there as Abstract Method
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """
        Generate a response from the environment based on the current messages.
        """
        raise NotImplementedError("env_response is not implemented for this environment.")

    def step(
        self,
        states: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        live_indices = [i for i, s in enumerate(states) if not s.get("completed", False)] # Use .get for safety
        if not live_indices:
            return states # All states are completed

        messages_to_step = []
        add_gen_prompt_flags = []
        continue_final_message_flags = []

        for i in live_indices:
            current_messages = states[i]["messages"]
            messages_to_step.append(current_messages)
            # Determine if we are starting or continuing the assistant message
        if states[live_indices[0]]["messages"][-1]["role"] == "user":
            add_gen_prompt=True
            continue_final_message=False
        else:
            add_gen_prompt=False
            continue_final_message=True

        if debug:
            if live_indices and not states[live_indices[0]].get("completed", False):
                print("-" * 30 + f" Step Start (Live: {len(live_indices)}) " + "-" * 30)
                print(f"State {live_indices[0]} Input Messages (last 2): {messages_to_step[0][-2:]}")
                # time.sleep(1)

        # --- Call LLM ---
        # Note: vLLM/TRL might need specific handling for batching requests with mixed
        # add_generation_prompt and continue_final_message flags.
        # Assuming the llm.chat handles this correctly or processes sequentially if needed.
        # If batching fails, process one by one.
        try:
             llm_responses = llm.chat(
                 messages_to_step,
                 sampling_params=sampling_params,
                 use_tqdm=False,
                 add_generation_prompt=add_gen_prompt,
                 continue_final_message=continue_final_message,
             ) # type: ignore
        except TypeError as e:
             # Fallback to processing one by one if batching flags fails
             logger.warning(f"LLM chat might not support list flags ({e}). Processing sequentially.")
             llm_responses = []
             for idx, msg_list in enumerate(messages_to_step):
                 response = llm.chat(
                     [msg_list], # Send as list containing one conversation
                     sampling_params=sampling_params,
                     use_tqdm=False,
                     add_generation_prompt=add_gen_prompt_flags[idx],
                     continue_final_message=continue_final_message_flags[idx],
                 )[0] # Get the single response from the list
                 llm_responses.append(response)


        # --- Process LLM Responses ---
        for i, live_idx in enumerate(live_indices):
            state = states[live_idx]
            llm_response_obj = llm_responses[i]
            new_response_text = llm_response_obj.outputs[0].text
            new_response_ids = llm_response_obj.outputs[0].token_ids

            if debug and live_idx == 0:
                print(f"State {live_idx} LLM Output Segment: '{new_response_text}'")
                # time.sleep(1)

            # Initialize state fields if first step
            if "completion_ids" not in state: state["completion_ids"] = []
            if "completion_mask" not in state: state["completion_mask"] = []
            if "prompt_ids" not in state: state["prompt_ids"] = []
            if "successful_func_calls" not in state: state["successful_func_calls"] = []

            # Store prompt IDs only once at the beginning
            if not state["prompt_ids"] and llm_response_obj.prompt_token_ids:
                state["prompt_ids"] = list(llm_response_obj.prompt_token_ids)

            # Update assistant message content
            if state["messages"][-1]["role"] == "user":
                # Start new assistant message
                state["messages"].append({"role": "assistant", "content": new_response_text})
            elif state["messages"][-1]["role"] == "assistant":
                # Append to existing assistant message
                state["messages"][-1]["content"] += new_response_text
            else:
                 logger.error(f"State {live_idx}: Cannot append response, last message role is {state['messages'][-1]['role']}")
                 state["completed"] = True # Mark as completed due to error state
                 state["error_message"] = "Internal error: Unexpected message structure."
                 continue # Skip further processing for this state

            # Update completion IDs and mask for the *generated* part
            state["completion_ids"].extend(new_response_ids)
            state["completion_mask"].extend([1] * len(new_response_ids)) # Generated tokens are unmasked (1)

            # --- Check for Stop Tokens and Handle Actions ---
            current_content = state["messages"][-1]["content"]
            completed_this_step = False
            stop_token_found = None

            if current_content.endswith("</tool>"):
                stop_token_found = "</tool>"
                if debug and live_idx == 0: print(f"State {live_idx}: Detected </tool>")
                # --- Handle Tool Call ---
                try:
                    parsed = self.llm_parser.parse(current_content)
                    if hasattr(parsed, "tool") and parsed.tool is not None:
                        tool_result_json, state = self.call_tool(parsed.tool, state=state, debug=debug)
                        tool_result_str = f"<tool_result> {tool_result_json} </tool_result>"

                        if debug and live_idx == 0: print(f"State {live_idx}: Tool Result: {tool_result_str}")

                        # Append result to assistant message
                        state["messages"][-1]["content"] += tool_result_str

                        # Tokenize result and update state (masking the result)
                        # Use add_special_tokens=False to avoid extra BOS/EOS here
                        result_tokens = self.tokenizer.encode(tool_result_str, add_special_tokens=False)
                        state["completion_ids"].extend(result_tokens)
                        state["completion_mask"].extend([self.env_mask] * len(result_tokens)) # Mask injected tokens

                        # Check if max tool interactions reached AFTER this call
                        tool_interactions = self._get_tool_interaction_count(state["messages"])
                        if tool_interactions >= self.max_steps_per_turn:
                             logger.warning(f"State {live_idx}: Reached max tool interactions ({self.max_steps_per_turn}) after tool call.")
                             # Append TASK_ERROR to signify forced stop? Or just complete?
                             # Let's just mark completed for now. The reward can penalize this.
                             state["messages"][-1]["content"] += "\n<TASK_ERROR> Max tool interactions reached.</TASK_ERROR>"
                             error_tokens = self.tokenizer.encode("\n<TASK_ERROR> Max tool interactions reached.</TASK_ERROR>", add_special_tokens=False)
                             state["completion_ids"].extend(error_tokens)
                             state["completion_mask"].extend([1] * len(error_tokens)) # Model didn't generate, but part of final state
                             completed_this_step = True
                        else:
                             # Continue generation in the next step
                             state["completed"] = False
                             completed_this_step = False # Explicitly not completed

                    else:
                        logger.warning(f"State {live_idx}: Found </tool> but failed to parse tool command from content.")
                        error_str = "<tool_result> Error: Could not parse tool command after </tool> tag. </tool_result>"
                        state["messages"][-1]["content"] += error_str
                        error_tokens = self.tokenizer.encode(error_str, add_special_tokens=False)
                        state["completion_ids"].extend(error_tokens)
                        state["completion_mask"].extend([self.env_mask] * len(error_tokens))
                        # Continue generation, hoping the model corrects or finishes
                        state["completed"] = False
                        completed_this_step = False

                except Exception as e:
                    logger.exception(f"State {live_idx}: Error during tool call processing: {e}")
                    error_str = f"<tool_result> Error: Exception during tool processing: {e} </tool_result>"
                    state["messages"][-1]["content"] += error_str
                    error_tokens = self.tokenizer.encode(error_str, add_special_tokens=False)
                    state["completion_ids"].extend(error_tokens)
                    state["completion_mask"].extend([self.env_mask] * len(error_tokens))
                    # Mark as completed with error
                    state["messages"][-1]["content"] += "\n<TASK_ERROR> Internal error during tool execution.</TASK_ERROR>"
                    error_tokens_term = self.tokenizer.encode("\n<TASK_ERROR> Internal error during tool execution.</TASK_ERROR>", add_special_tokens=False)
                    state["completion_ids"].extend(error_tokens_term)
                    state["completion_mask"].extend([1] * len(error_tokens_term))
                    state["completed"] = True
                    completed_this_step = True

            elif current_content.endswith("<TASK_FINISHED>"):
                stop_token_found = "<TASK_FINISHED>"
                if debug and live_idx == 0: print(f"State {live_idx}: Detected <TASK_FINISHED>")
                state["completed"] = True
                completed_this_step = True
            elif current_content.endswith("<TASK_ERROR>"):
                stop_token_found = "<TASK_ERROR>"
                if debug and live_idx == 0: print(f"State {live_idx}: Detected <TASK_ERROR>")
                state["completed"] = True
                completed_this_step = True

            # Check max tokens completion
            if not completed_this_step and len(state["completion_ids"]) >= sampling_params.max_tokens:
                 logger.warning(f"State {live_idx}: Reached max tokens ({sampling_params.max_tokens}). Truncating and marking completed.")
                 state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                 state["completion_mask"] = state["completion_mask"][:sampling_params.max_tokens]
                 # Append TASK_ERROR? Maybe just truncate is better.
                 # Let's add a note in the content
                 state["messages"][-1]["content"] += "\n<TASK_ERROR> Max tokens reached during generation.</TASK_ERROR>"
                 state["completed"] = True
                 completed_this_step = True

            # --- Final State Updates for Completed Entries ---
            if completed_this_step:
                 if debug and live_idx == 0: print(f"State {live_idx}: Marked as completed.")

                 # Ensure mask and ids lengths match after potential truncation/error append
                 min_len = min(len(state["completion_ids"]), len(state["completion_mask"]))
                 state["completion_ids"] = state["completion_ids"][:min_len]
                 state["completion_mask"] = state["completion_mask"][:min_len]

                 # --- Execute Ground Truth Calls (for final state comparison in reward) ---
                 # This happens *after* the model's generation is finished.
                 # It updates the 'ground_truth_environment' based on the 'answer' field.
                 try:
                      if "answer" in state["dataset_row"] and state["dataset_row"]["answer"]:
                           if debug and live_idx == 0: print(f"State {live_idx}: Executing ground truth calls for final state...")
                           _, state = self.call_tool(
                               tool_json=None, # Not used for GT
                               state=state,
                               debug=(debug and (live_idx == 0)),
                               ground_truth=True,
                           )
                           if debug and live_idx == 0: print(f"State {live_idx}: Ground truth execution finished.")
                      else:
                           if debug and live_idx == 0: print(f"State {live_idx}: No ground truth answer found to execute.")

                 except Exception as e:
                      logger.error(f"State {live_idx}: Failed to execute ground truth calls: {e}")
                      # Log this error, but don't necessarily stop the whole process
                      state["ground_truth_error"] = str(e)

                 # Final check for length consistency (paranoid check)
                 if len(state["completion_mask"]) != len(state["completion_ids"]):
                     logger.error(f"State {live_idx}: Mismatch after completion! Mask: {len(state['completion_mask'])}, IDs: {len(state['completion_ids'])}")
                     min_len = min(len(state["completion_mask"]), len(state["completion_ids"]))
                     state["completion_mask"] = state["completion_mask"][:min_len]
                     state["completion_ids"] = state["completion_ids"][:min_len]

            # Assert length consistency before next loop/end
            assert len(state["completion_mask"]) == len(state["completion_ids"]), \
                f"State {live_idx}: Final length mismatch! Mask: {len(state['completion_mask'])}, IDs: {len(state['completion_ids'])}"

        if debug: print("-" * 30 + f" Step End (Live: {len(live_indices)}) " + "-" * 30)
        return states


    def _initialize_environments(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Initialize environment instances for the given states (Unchanged)"""
        # Ensure tokenizer is available if needed within class init, though unlikely
        for i in range(len(states)):
            if "instance_id" not in states[i]:
                states[i]["instance_id"] = id(states[i]) # Simple instance ID for this batch
            instance_id = states[i]["instance_id"]

            # Check if already initialized for this instance ID in this run
            if instance_id in self.env_instances:
                 # Ensure state dict points to the correct instances
                 states[i]["environment"] = self.env_instances[instance_id]["main"]
                 states[i]["ground_truth_environment"] = self.env_instances[instance_id]["ground_truth"]
                 states[i]["initial_environment"] = self.env_instances[instance_id]["initial_instance"]
                 continue # Already initialized

            # Initialize if not found
            self.env_instances[instance_id] = {"main": {}, "ground_truth": {}, "initial_instance": {}}
            states[i]["environment"] = self.env_instances[instance_id]["main"]
            states[i]["ground_truth_environment"] = self.env_instances[instance_id]["ground_truth"]
            states[i]["initial_environment"] = self.env_instances[instance_id]["initial_instance"]


            involved_classes = states[i]["dataset_row"]["involved_classes"]
            initial_config_str = states[i]["dataset_row"].get("initial_config", "{}")
            try:
                initial_config = json.loads(initial_config_str)
            except json.JSONDecodeError:
                logger.error(f"State {i}: Invalid JSON in initial_config: {initial_config_str}")
                initial_config = {} # Use empty config on error

            for class_name in involved_classes:
                 try:
                     module_name = self.CLASS_FILE_PATH_MAPPING[class_name]
                     module = importlib.import_module(module_name)
                     class_ = getattr(module, class_name)

                     # Create instances
                     class_instance = class_()
                     ground_truth_class_instance = class_()
                     initial_instance_copy = class_()

                     # Configure non-stateless classes
                     if class_name not in self.STATELESS_CLASSES:
                         class_initial_config = initial_config.get(class_name, {})
                         # Use deepcopy to avoid shared state issues across instances
                         class_instance._load_scenario(copy.deepcopy(class_initial_config))
                         ground_truth_class_instance._load_scenario(copy.deepcopy(class_initial_config))
                         initial_instance_copy._load_scenario(copy.deepcopy(class_initial_config))

                     # Store instances in state and central dict
                     states[i]["environment"][class_name] = class_instance
                     states[i]["ground_truth_environment"][class_name] = ground_truth_class_instance
                     states[i]["initial_environment"][class_name] = initial_instance_copy

                 except KeyError:
                      logger.error(f"State {i}: Class '{class_name}' not found in CLASS_FILE_PATH_MAPPING.")
                      # Handle error appropriately, maybe skip this class or raise
                 except ImportError:
                      logger.error(f"State {i}: Could not import module for class '{class_name}'.")
                 except AttributeError:
                      logger.error(f"State {i}: Could not find class '{class_name}' in its module.")
                 except Exception as e:
                      logger.exception(f"State {i}: Error initializing environment for class '{class_name}': {e}")

        return states


    def cleanup_instances(self) -> None:
        """Clean up all environment instances"""
        if self.env_instances:
             logger.debug(f"Cleaning up {len(self.env_instances)} environment instance sets.")
        self.env_instances.clear()
        gc.collect()

    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        dataset_rows: List[Dict[str, Any]] = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        # Apply env-specific sampling args (stop tokens etc.)
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)
        logger.info(f"Using sampling params: stop={custom_sp.stop}, include_stop={custom_sp.include_stop_str_in_output}, max_tokens={custom_sp.max_tokens}")

        if dataset_rows is None or len(prompts) != len(dataset_rows):
            raise ValueError("dataset_rows must be provided and match length of prompts")

        # Initialize state variables
        states = []
        for i, m in enumerate(prompts):
             initial_state = {
                 "messages": copy.deepcopy(m), # Starts with user message
                 # "multi_turn_history": copy.deepcopy(m), # Removed for simplicity
                 "prompt_messages_count": len(m), # Should be 1 (user message)
                 "prompt_ids": [],
                 "completed": False,
                 "completion_ids": [],
                 "completion_mask": [],
                 "dataset_row": copy.deepcopy(dataset_rows[i]),
                 "successful_func_calls": [], # Initialize list for successful calls
                 "id": dataset_rows[i].get("id", f"unidentified_{i}") # Add ID for logging
             }
             states.append(initial_state)


        if debug:
            print(f"Initial User Prompt (State 0): {states[0]['messages'][0]['content'][:500]}...") # Print start of prompt
            print(f"Number of Rollouts: {len(states)}")
            # time.sleep(1)

        # Initialize environments for the batch
        try:
             states = self._initialize_environments(states)
        except Exception as e:
             logger.exception(f"Fatal error during environment initialization: {e}")
             # Mark all states as errored?
             for s in states:
                 s["completed"] = True
                 s["error_message"] = f"Environment initialization failed: {e}"
             # Return immediately with error state
             return self._package_generate_output(states)


        # Main generation loop
        step_count = 0
        max_overall_steps = self.max_steps_per_turn * 5 # Safety break: max interactions * buffer
        all_completed = False
        while not all_completed and step_count < max_overall_steps:
            step_count += 1
            if debug: print(f"\n--- Starting Generation Step {step_count} ---")
            try:
                 states = self.step(states, llm, custom_sp, debug=debug)
            except Exception as e:
                 logger.exception(f"Fatal error during generation step {step_count}: {e}")
                 # Mark all live states as errored
                 for s in states:
                     if not s.get("completed", False):
                         s["completed"] = True
                         s["error_message"] = f"Generation failed at step {step_count}: {e}"
                 break # Exit loop on fatal error

            all_completed = all(s.get("completed", False) for s in states)
            if debug: print(f"--- Finished Generation Step {step_count} (All Completed: {all_completed}) ---")
            # Optional: Add a small sleep if debugging to make logs readable
            # if debug: time.sleep(0.5)

        if step_count >= max_overall_steps:
             logger.warning(f"Generation stopped after reaching max overall steps ({max_overall_steps}).")
             # Mark any non-completed states as errored/incomplete
             for s in states:
                 if not s.get("completed", False):
                     s["completed"] = True
                     s["error_message"] = "Max overall generation steps reached."
                     # Optionally append TASK_ERROR to content if needed
                     if s["messages"][-1]["role"] == "assistant":
                          s["messages"][-1]["content"] += "\n<TASK_ERROR> Max steps reached.</TASK_ERROR>"


        # Cleanup environments after batch is fully processed
        self.cleanup_instances()

        # Package results
        output = self._package_generate_output(states)
        return output

    def _package_generate_output(self, states: List[Dict[str, Any]]) -> Dict:
         """Helper to format the final output of the generate method."""
         completion_messages = []
         for s in states:
             # Extract only the assistant message(s) - should be just one now
             assistant_msgs = [msg for msg in s["messages"][s["prompt_messages_count"]:] if msg["role"] == "assistant"]
             completion_messages.append(assistant_msgs)

         completion_ids = [s.get("completion_ids", []) for s in states]
         completion_mask = [s.get("completion_mask", []) for s in states]

         # Final length check before returning
         for i in range(len(states)):
              if len(completion_ids[i]) != len(completion_mask[i]):
                   logger.error(f"Final output length mismatch for state {i}! IDs: {len(completion_ids[i])}, Mask: {len(completion_mask[i])}. Truncating.")
                   min_len = min(len(completion_ids[i]), len(completion_mask[i]))
                   completion_ids[i] = completion_ids[i][:min_len]
                   completion_mask[i] = completion_mask[i][:min_len]


         output = {
             "ids": completion_ids,
             "messages": completion_messages, # List of lists, each inner list has one assistant dict
             "mask": completion_mask,
             "states": states, # Return the final states for reward calculation etc.
         }
         return output

    def eval(self, model: Union[str, LLM], batch_size: int = 10, **kwargs: Any):
        # Evaluation logic likely happens in the Trainer, this is just a placeholder
        logger.warning("BfclITEnv.eval is basic. Evaluation typically handled by Trainer.")
        if self.eval_dataset is None:
            self.eval_dataset = self.get_eval_dataset(max_num_turns=self.max_num_turns) # Use property

        # Basic structure, actual reward calculation/logging is in Trainer
        rewards = [] # Placeholder
        # Example: Iterate through eval_dataset, call generate, calculate rewards
        # for i in range(0, len(self.eval_dataset), batch_size):
        #     batch = self.eval_dataset[i:i+batch_size]
        #     prompts = [item['prompt'] for item in batch]
        #     dataset_rows = list(batch) # Pass full row data
        #     # Need LLM instance here
        #     # results = self.generate(prompts, llm, sampling_params, dataset_rows)
        #     # Calculate rewards based on results['states'] and rubric
        #     # rewards.extend(...)
        logger.info(f"Eval dataset size: {len(self.eval_dataset)}. Returning dataset and empty rewards.")
        return self.eval_dataset, rewards