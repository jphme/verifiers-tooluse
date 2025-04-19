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

# Revised Prompt
BFCL_INTHINKING_USER_PROMPT = """You are an expert assistant that reasons step-by-step and uses tools to answer user questions.
You have access to the following tools:

{tools}

Follow this precise workflow:
1.  **Think Step-by-Step:** Analyze the user query and devise a plan. Break down the problem into smaller steps. Explain your reasoning within `<think>` tags.
2.  **Identify Tool Need:** Determine if a tool call is necessary for the current step of your plan.
3.  **Execute Tool Call (If Needed):**
    *   If you need to use tools, formulate ONE or MORE tool calls required for the *current step* as a JSON list within `<tool>` tags.
    *   Each item in the list MUST be a JSON object with "name" (string) and "args" (dictionary) keys.
    *   Example: `<tool> [{{"name": "get_user_id", "args": {{"user": "Alice"}}}}, {{"name": "search_messages", "args": {{"keyword": "urgent"}}}}] </tool>`
    *   **CRITICAL:** Only use the provided tools and their exact argument names and types. Adhere strictly to the JSON list format. Do NOT include tool descriptions or schemas within the `<tool>` tags.
    *   After outputting the `<tool>` tag, STOP your generation. The system will execute the tools and provide results.
4.  **Receive Tool Results:** Tool results will be provided back to you enclosed in `<tool_result>` tags.
5.  **Analyze Results & Continue Thinking:**
    *   Carefully analyze the information provided in the `<tool_result>`.
    *   Continue your thought process within `<think>` tags. Explain how the results inform your plan and what the next step should be.
    *   If more tool calls are needed to complete the task, go back to step 3.
6.  **Provide Final Answer:**
    *   Once you have gathered all necessary information and completed the plan, conclude your thinking with `</think>`.
    *   Provide a final, comprehensive answer or summary to the user.
    *   End your *entire* response with `<TASK_FINISHED>`.
7.  **Handle Errors:**
    *   If you encounter an unrecoverable error (e.g., a required tool fails consistently, the task is impossible with available tools), explain the issue within `<think>` tags, conclude with `</think>`, state the problem clearly, and end your *entire* response with `<TASK_ERROR>`.
    *   If a tool call fails but you think you can recover (e.g., by trying different arguments), explain this in your thinking and proceed accordingly.

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

# In bfcl_inthinking_env.py

    def step(
        self,
        states: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        live_indices = [i for i, s in enumerate(states) if not s.get("completed", False)]
        if not live_indices:
            return states

        # --- Separate states based on required LLM call type ---
        start_states_data = [] # { "index": original_index, "messages": messages }
        continue_states_data = [] # { "index": original_index, "messages": messages }

        for i in live_indices:
            state = states[i]
            if state.get("completed", False):
                continue

            current_messages = state["messages"]
            # Check the role of the actual last message to decide the call type
            if current_messages[-1]["role"] == "user":
                start_states_data.append({"index": i, "messages": current_messages})
            elif current_messages[-1]["role"] == "assistant":
                continue_states_data.append({"index": i, "messages": current_messages})
            else:
                # This case should ideally not happen if state logic is correct
                logger.error(f"State {i}: Unexpected last message role {current_messages[-1]['role']} in live state. Marking as error.")
                state["completed"] = True
                state["error_message"] = "Internal error: Invalid message structure before LLM call."

        # --- Perform LLM Calls Separately ---
        all_llm_responses = {} # Store responses mapped by original index: { index: llm_response_obj }

        # Call for states starting a new message
        if start_states_data:
            start_indices = [d["index"] for d in start_states_data]
            start_messages = [d["messages"] for d in start_states_data]
            if debug: print(f"---> Calling LLM for {len(start_indices)} states (add_generation_prompt=True)")
            try:
                start_responses = llm.chat(
                    start_messages, sampling_params=sampling_params, use_tqdm=False,
                    add_generation_prompt=True,
                    continue_final_message=False, # Explicitly False
                )
                # Ensure the number of responses matches the number of prompts sent
                if len(start_responses) != len(start_indices):
                     logger.error(f"LLM call (start group) mismatch: Sent {len(start_indices)} prompts, received {len(start_responses)} responses.")
                     # Handle mismatch - mark corresponding states as error?
                     for original_index in start_indices:
                         if original_index not in all_llm_responses: # Avoid overwriting if already processed somehow
                            states[original_index]["completed"] = True
                            states[original_index]["error_message"] = "LLM response count mismatch (start group)."
                else:
                    # Map responses back to original indices
                    for idx, response_obj in enumerate(start_responses):
                        original_index = start_indices[idx]
                        all_llm_responses[original_index] = response_obj
            except Exception as e:
                logger.exception(f"Error during LLM chat call (start group): {e}")
                for original_index in start_indices:
                    states[original_index]["completed"] = True
                    states[original_index]["error_message"] = f"LLM chat call failed (start group): {e}"

        # Call for states continuing a message
        if continue_states_data:
            continue_indices = [d["index"] for d in continue_states_data]
            continue_messages = [d["messages"] for d in continue_states_data]
            if debug: print(f"---> Calling LLM for {len(continue_indices)} states (continue_final_message=True)")
            try:
                continue_responses = llm.chat(
                    continue_messages, sampling_params=sampling_params, use_tqdm=False,
                    add_generation_prompt=False, # Explicitly False
                    continue_final_message=True,
                )
                 # Ensure the number of responses matches the number of prompts sent
                if len(continue_responses) != len(continue_indices):
                     logger.error(f"LLM call (continue group) mismatch: Sent {len(continue_indices)} prompts, received {len(continue_responses)} responses.")
                     # Handle mismatch
                     for original_index in continue_indices:
                         if original_index not in all_llm_responses:
                            states[original_index]["completed"] = True
                            states[original_index]["error_message"] = "LLM response count mismatch (continue group)."
                else:
                    # Map responses back to original indices
                    for idx, response_obj in enumerate(continue_responses):
                        original_index = continue_indices[idx]
                        # Check if index already exists (shouldn't happen if logic is correct)
                        if original_index in all_llm_responses:
                             logger.warning(f"State {original_index} received response from both start and continue groups. Using continue response.")
                        all_llm_responses[original_index] = response_obj
            except Exception as e:
                logger.exception(f"Error during LLM chat call (continue group): {e}")
                for original_index in continue_indices:
                     # Avoid overwriting error from start group if it happened
                     if not states[original_index].get("completed", False):
                        states[original_index]["completed"] = True
                        states[original_index]["error_message"] = f"LLM chat call failed (continue group): {e}"

        # --- Process All Received LLM Responses ---
        if not all_llm_responses:
             if debug: print("No LLM responses received or processed in this step.")
             return states # Return states as they might have been updated with errors

        if debug: print(f"--- Processing {len(all_llm_responses)} LLM responses ---")

        # Iterate through the indices for which we expected and potentially received responses
        processed_indices = sorted(all_llm_responses.keys()) # Process in order
        for live_idx in processed_indices:
            # live_idx = llm_call_indices[i] # This was incorrect index mapping before
            state = states[live_idx]
            llm_response_obj = all_llm_responses[live_idx] # Get response using correct index

            # Skip processing if the state was marked completed due to an LLM error above
            if state.get("completed", False) and "LLM" in state.get("error_message", ""):
                continue
            if state.get("completed", False): # General check
                logger.warning(f"State {live_idx} was already completed before processing its LLM response. Skipping.")
                continue

            new_response_text = llm_response_obj.outputs[0].text
            new_response_ids = llm_response_obj.outputs[0].token_ids

            if debug and live_idx == 0: # Log for the first state processed in this batch
                print(f"State {live_idx} LLM Output Segment: '{new_response_text}'")

            # Initialize state fields if needed
            if "completion_ids" not in state: state["completion_ids"] = []
            if "completion_mask" not in state: state["completion_mask"] = []
            if "prompt_ids" not in state: state["prompt_ids"] = []
            if "successful_func_calls" not in state: state["successful_func_calls"] = []
            if not state["prompt_ids"] and llm_response_obj.prompt_token_ids:
                state["prompt_ids"] = list(llm_response_obj.prompt_token_ids)

            # Append new text segment
            # Determine if starting or continuing based on message history *before* the call
            # This relies on the separation logic being correct.
            last_message_role_before_call = None
            if live_idx in [d['index'] for d in start_states_data]:
                 last_message_role_before_call = 'user'
            elif live_idx in [d['index'] for d in continue_states_data]:
                 last_message_role_before_call = 'assistant'

            if last_message_role_before_call == 'user':
                state["messages"].append({"role": "assistant", "content": new_response_text})
            elif last_message_role_before_call == 'assistant':
                state["messages"][-1]["content"] += new_response_text
            else:
                 # This case should have been caught earlier, but handle defensively
                 logger.error(f"State {live_idx}: Could not determine if starting or continuing. Last role: {state['messages'][-1]['role']}. Marking error.")
                 state["completed"] = True
                 state["error_message"] = "Internal error: Ambiguous state for appending LLM response."
                 continue


            # Update completion IDs/mask for the *generated* part
            state["completion_ids"].extend(new_response_ids)
            state["completion_mask"].extend([1] * len(new_response_ids))

            # --- Check for Stop Conditions (using the logic from the previous correction) ---
            # (The rest of the logic for checking termination tokens, tool calls, max tokens,
            #  truncation, mask rebuilding, and final state updates remains the same as
            #  the previous answer - Paste that logic block here)
            # --- Start of pasted logic block ---
            current_content = state["messages"][-1]["content"] # Full accumulated content
            completed_this_step = False

            # 1. Check for TERMINATION tokens in the FULL accumulated content
            termination_token = None
            if "<TASK_FINISHED>" in current_content: termination_token = "<TASK_FINISHED>"
            elif "<TASK_ERROR>" in current_content: termination_token = "<TASK_ERROR>"

            if termination_token:
                if debug and live_idx == 0: print(f"State {live_idx}: Found termination token '{termination_token}' in accumulated content.")
                completed_this_step = True
                state["completed"] = True
                first_term_idx = current_content.find(termination_token)
                if first_term_idx != -1:
                    target_content_len = first_term_idx + len(termination_token)
                    if len(current_content) > target_content_len:
                        if debug and live_idx == 0: print(f"State {live_idx}: Truncating content after first '{termination_token}'.")
                        state["messages"][-1]["content"] = current_content[:target_content_len]
                        try:
                            full_sequence_tokens = self.tokenizer.encode(state["messages"][-1]["content"], add_special_tokens=False)
                            prompt_len = len(state.get("prompt_ids", []))
                            state["completion_ids"] = full_sequence_tokens[prompt_len:]
                            mask = []
                            completion_text_for_mask = self.tokenizer.decode(state["completion_ids"], skip_special_tokens=True)
                            segments = completion_text_for_mask.split("<tool_result>")
                            first_segment = True
                            for segment in segments:
                                if not first_segment and "</tool_result>" in segment:
                                    result_part, rest_part = segment.split("</tool_result>", 1)
                                    result_tokens_segment = self.tokenizer.encode("<tool_result>" + result_part + "</tool_result>", add_special_tokens=False)
                                    mask.extend([self.env_mask] * len(result_tokens_segment))
                                    rest_tokens_segment = self.tokenizer.encode(rest_part, add_special_tokens=False)
                                    mask.extend([1] * len(rest_tokens_segment))
                                else:
                                    part_tokens = self.tokenizer.encode(segment, add_special_tokens=False)
                                    if part_tokens: mask.extend([1] * len(part_tokens))
                                first_segment = False
                            state["completion_mask"] = mask[:len(state["completion_ids"])]
                            if len(state["completion_mask"]) != len(state["completion_ids"]):
                                logger.warning(f"State {live_idx}: Mask length mismatch after rebuild ({len(state['completion_mask'])} vs {len(state['completion_ids'])}). Adjusting.")
                                min_len_rebuild = min(len(state["completion_ids"]), len(state["completion_mask"]))
                                state["completion_ids"] = state["completion_ids"][:min_len_rebuild]
                                state["completion_mask"] = state["completion_mask"][:min_len_rebuild]
                        except Exception as e_recalc:
                             logger.exception(f"State {live_idx}: Error recalculating tokens/mask after truncation: {e_recalc}.")

            # 2. Check for Tool Call Stop Token (only if not already terminated)
            elif not completed_this_step and current_content.endswith("</tool>"):
                stop_token_found = "</tool>"
                if debug and live_idx == 0: print(f"State {live_idx}: Detected </tool>")
                try:
                    last_tool_start_idx = current_content.rfind("<tool>")
                    if last_tool_start_idx != -1:
                        last_tool_segment = current_content[last_tool_start_idx:]
                        parsed = self.llm_parser.parse(last_tool_segment)
                        if hasattr(parsed, "tool") and parsed.tool is not None:
                            tool_result_json, state = self.call_tool(parsed.tool, state=state, debug=debug)
                            tool_result_str = f"<tool_result> {tool_result_json} </tool_result>"
                            state["messages"][-1]["content"] += tool_result_str
                            result_tokens = self.tokenizer.encode(tool_result_str, add_special_tokens=False)
                            state["completion_ids"].extend(result_tokens)
                            state["completion_mask"].extend([self.env_mask] * len(result_tokens))
                            tool_interactions = self._get_tool_interaction_count(state["messages"])
                            if tool_interactions >= self.max_steps_per_turn:
                                logger.warning(f"State {live_idx}: Reached max tool interactions ({self.max_steps_per_turn}).")
                                error_marker = "\n<TASK_ERROR> Max tool interactions reached.</TASK_ERROR>"
                                state["messages"][-1]["content"] += error_marker
                                error_tokens = self.tokenizer.encode(error_marker, add_special_tokens=False)
                                state["completion_ids"].extend(error_tokens)
                                state["completion_mask"].extend([1] * len(error_tokens))
                                completed_this_step = True; state["completed"] = True
                            else:
                                completed_this_step = False; state["completed"] = False
                        else:
                            logger.warning(f"State {live_idx}: Failed to parse tool from last segment.")
                            error_str = "<tool_result> Error: Could not parse recent tool command. </tool_result>"
                            state["messages"][-1]["content"] += error_str
                            error_tokens = self.tokenizer.encode(error_str, add_special_tokens=False)
                            state["completion_ids"].extend(error_tokens)
                            state["completion_mask"].extend([self.env_mask] * len(error_tokens))
                            completed_this_step = False; state["completed"] = False
                    else:
                        logger.warning(f"State {live_idx}: Found </tool> without preceding <tool>.")
                        error_str = "<tool_result> Error: Malformed output - stray </tool>. </tool_result>"
                        state["messages"][-1]["content"] += error_str
                        error_tokens = self.tokenizer.encode(error_str, add_special_tokens=False)
                        state["completion_ids"].extend(error_tokens)
                        state["completion_mask"].extend([self.env_mask] * len(error_tokens))
                        completed_this_step = False; state["completed"] = False
                except Exception as e:
                    logger.exception(f"State {live_idx}: Error during tool call processing: {e}")
                    error_str = f"<tool_result> Error: Exception during tool processing: {e} </tool_result>"
                    state["messages"][-1]["content"] += error_str
                    error_tokens = self.tokenizer.encode(error_str, add_special_tokens=False)
                    state["completion_ids"].extend(error_tokens)
                    state["completion_mask"].extend([self.env_mask] * len(error_tokens))
                    error_marker = "\n<TASK_ERROR> Internal error during tool execution.</TASK_ERROR>"
                    state["messages"][-1]["content"] += error_marker
                    error_tokens_term = self.tokenizer.encode(error_marker, add_special_tokens=False)
                    state["completion_ids"].extend(error_tokens_term)
                    state["completion_mask"].extend([1] * len(error_tokens_term))
                    completed_this_step = True; state["completed"] = True

            # 3. Check max tokens (only if not already completed)
            # Use state.get("completion_ids", []) for safety in case it wasn't initialized
            current_completion_len = len(state.get("completion_ids", []))
            if not completed_this_step and current_completion_len >= sampling_params.max_tokens:
                 logger.warning(f"State {live_idx}: Reached max tokens ({sampling_params.max_tokens}). Current: {current_completion_len}")
                 state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                 state["completion_mask"] = state["completion_mask"][:sampling_params.max_tokens]
                 prompt_toks = state.get("prompt_ids", [])
                 full_toks_truncated = prompt_toks + state["completion_ids"]
                 # Decode carefully, potentially without skipping special tokens if needed by template structure
                 try:
                      # Attempt decoding, might fail if tokens are invalid/incomplete
                      decoded_content = self.tokenizer.decode(full_toks_truncated, skip_special_tokens=True)
                      state["messages"][-1]["content"] = decoded_content
                 except Exception as e_decode:
                      logger.error(f"State {live_idx}: Failed to decode truncated tokens: {e_decode}. Content might be inconsistent.")
                      # Fallback: Keep content as is before truncation attempt, just mark completed
                 if "<TASK_FINISHED>" not in state["messages"][-1]["content"] and "<TASK_ERROR>" not in state["messages"][-1]["content"]:
                      state["messages"][-1]["content"] += "\n<TASK_ERROR> Max tokens reached.</TASK_ERROR>"
                 state["completed"] = True
                 completed_this_step = True

            # --- Final State Updates (only run if completed *in this step*) ---
            if completed_this_step:
                 if debug and live_idx == 0: print(f"State {live_idx}: Marked as completed in this step.")
                 min_len = min(len(state["completion_ids"]), len(state["completion_mask"]))
                 if len(state["completion_ids"]) != min_len or len(state["completion_mask"]) != min_len:
                      logger.warning(f"State {live_idx}: Correcting length mismatch before GT ({len(state['completion_ids'])} vs {len(state['completion_mask'])}).")
                      state["completion_ids"] = state["completion_ids"][:min_len]
                      state["completion_mask"] = state["completion_mask"][:min_len]
                 try:
                      if "answer" in state["dataset_row"] and state["dataset_row"]["answer"]:
                           if debug and live_idx == 0: print(f"State {live_idx}: Executing ground truth calls...")
                           _, state = self.call_tool(tool_json=None, state=state, debug=(debug and (live_idx == 0)), ground_truth=True)
                           if debug and live_idx == 0: print(f"State {live_idx}: Ground truth execution finished.")
                 except Exception as e:
                      logger.error(f"State {live_idx}: Failed GT calls: {e}")
                      state["ground_truth_error"] = str(e)
                 assert len(state["completion_mask"]) == len(state["completion_ids"]), \
                     f"State {live_idx}: Final length mismatch! Mask: {len(state['completion_mask'])}, IDs: {len(state['completion_ids'])}"
            # --- End of pasted logic block ---


        if debug: print("-" * 30 + f" Step End (Processed {len(all_llm_responses)} LLM responses) " + "-" * 30)
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