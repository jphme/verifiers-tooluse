# bfcl_inthinking_env.py

import copy
import gc
import importlib
import inspect
import json
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
# import torch # No longer needed directly here
from datasets import (
    Dataset,
    DatasetDict,  # type: ignore
)
from huanzhi_utils import load_file
from loguru import logger
from sklearn.model_selection import train_test_split
from trl.trainer.grpo_trainer import RewardFunc
# from transformers import PreTrainedTokenizerBase # No longer needed directly here

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.envs.tool_env import infer_schema_from_function
from verifiers.parsers import XMLParser
from verifiers.rubrics.bfcl_inthinking_rubric import BfclITRubric
from verifiers.tools.bfcl_tools import (
    INVOLVED_CLASS_TO_FUNC_DOC_PATH,
    construct_tools_from_involved_classes,
)

from ..imports import LLM, SamplingParams  # type: ignore

# Revised Prompt for Single Tool Call and Clarity
BFCL_INTHINKING_USER_PROMPT = """You are an expert assistant that reasons step-by-step and uses tools to answer user questions.

# Available tools:
{tools}

# Instructions:
Follow this precise workflow:
1.  **Think Step-by-Step:** Analyze the user query and devise a plan. Break down the problem into smaller steps. Explain your reasoning.
2.  **Identify Tool Need:** Determine if a tool call is necessary for the *current* step of your plan.
3.  **Execute Tool Call (If Needed):**
    * If you need to use ONE tool, formulate the SINGLE tool call as a JSON object within `<tool>` tags.
    * The JSON object MUST have "name" (string) and "args" (dictionary) keys.
    * Example: `<tool>{{"name": "get_user_id", "args": {{"user": "Alice"}}}}</tool>`
    * **CRITICAL:** Only use the provided tools and their exact argument names and types. Adhere strictly to the JSON object format. Do NOT include tool descriptions, schemas, or lists within the `<tool>` tags.
    * **IMPORTANT**: Use the tools WITHIN your thinking and planning phase. Only exit the reasoning phase with "</think>" if you have completed the task!
4.  **Receive Tool Results:** Tool results will be provided back to you enclosed in `<tool_result>` tags.
5.  **Analyze Results & Continue Thinking:**
    * Carefully analyze the information provided in the `<tool_result>`. Check if the call succeeded or failed.
    * Continue your thought process. Explain how the results (or errors) inform your plan and what the next step should be. If a tool failed, analyze the error and decide if you can retry with different arguments or need to take a different approach.
    * If another tool call is needed to complete the task, go back to step 3.
6.  **Provide Final Answer:**
    * Once you have gathered all necessary information and completed the plan, conclude your thinking with `</think>`.
    * Provide a final, comprehensive answer or summary to the user.
    * End your *entire* response with `<TASK_FINISHED>`.
7.  **Handle Errors:**
    * If you encounter an unrecoverable error (e.g., a required tool fails consistently, the task is impossible), conclude with `</think>`, state the problem clearly, and end your *entire* response with `<TASK_ERROR>`.

Remember, these are the available tools: {available_tool_names}

# User Question:
{user_query}"""

# Updated Few-Shot Example for Single Tool Call
FEW_SHOT_EXAMPLE = [
    {
        "role": "user",
        "content": BFCL_INTHINKING_USER_PROMPT.format(
            tools="""{"name": "get_user_id", "description": "Get user ID from user name.", "parameters": {"type": "dict", "properties": {"user": {"type": "string"}}, "required": ["user"]}, "response": {"type": "dict", "properties": {"user_id": {"type": "string"}}}}
{"name": "send_message", "description": "Send a message to a user.", "parameters": {"type": "dict", "properties": {"receiver_id": {"type": "string"}, "message": {"type": "string"}}, "required": ["receiver_id", "message"]}, "response": {"type": "dict", "properties": {"sent_status": {"type": "boolean"}, "message_id": {"type": "integer"}}}}""",
            available_tool_names="get_user_id, send_message", # Example available tools list
            user_query="Can you send a welcome message to the new user Bob?"
        )
    },
    {
        "role": "assistant",
        "content": """<think>
The user wants me to send a welcome message to a user named "Bob".
To use the `send_message` tool, I need the `receiver_id` for Bob.
I don't have Bob's ID yet.
My plan is:
1. Use the `get_user_id` tool to find Bob's user ID.
2. Use the `send_message` tool with the obtained ID and a welcome message.
First step is to get the user ID.
<tool>{"name": "get_user_id", "args": {"user": "Bob"}}</tool><tool_result>"Function Call {'name': 'get_user_id', 'args': {'user': 'Bob'}} Succeeded. Result: {'user_id': 'USR123'}"</tool_result>
Okay, the `get_user_id` tool successfully returned Bob's user ID as 'USR123'.
Now I have the `receiver_id` needed for the next step.
My plan was to send a welcome message. I will now use the `send_message` tool.
I'll compose a simple welcome message.
<tool>{"name": "send_message", "args": {"receiver_id": "USR123", "message": "Welcome to the team, Bob!"}}</tool><tool_result>"Function Call {'name': 'send_message', 'args': {'receiver_id': 'USR123', 'message': 'Welcome to the team, Bob!'}} Succeeded. Result: {'sent_status': True, 'message_id': 500}"</tool_result>
The `send_message` tool confirmed that the message was sent successfully (sent_status: True).
I have now completed all the steps required by the user's request.
I should inform the user that the message has been sent.
</think>
I have sent a welcome message to Bob (User ID: USR123).
<TASK_FINISHED>"""
    }
]

# Helper to get just tool names
def _get_tool_names_from_involved_classes(involved_classes: List[str]) -> str:
    """Helper to create a comma-separated string of available tool names."""
    available_tools = []
    if not involved_classes:
        return "None"
    try:
        for class_name in involved_classes:
            func_doc_path = INVOLVED_CLASS_TO_FUNC_DOC_PATH.get(class_name)
            if func_doc_path:
                func_doc = load_file(func_doc_path)
                for func in func_doc:
                    available_tools.append(func["name"])
        return ", ".join(sorted(list(set(available_tools)))) if available_tools else "None"
    except Exception as e:
        logger.warning(f"Could not retrieve available tool names: {e}")
        return "Error retrieving tools"

# Modify format_bfcl_prompt to include available tool names
def format_bfcl_prompt(
    involved_classes: List[str] | None = None,
    user_question: str | None = None,
    include_few_shot: bool = False, # Add flag
) -> List[Dict[str, str]]:
    messages = []
    if include_few_shot:
        # IMPORTANT: Deep copy the few-shot example to avoid modification issues
        messages.extend(copy.deepcopy(FEW_SHOT_EXAMPLE))

    # Construct the current query prompt
    tools_schema = construct_tools_from_involved_classes(involved_classes)
    available_tool_names = _get_tool_names_from_involved_classes(involved_classes) # Get tool names

    current_query_message = {
        "role": "user",
        "content": BFCL_INTHINKING_USER_PROMPT.format(
            tools=tools_schema,
            available_tool_names=available_tool_names, # Pass tool names to prompt
            user_query=user_question
        ),
    }

    # Append the actual query *after* the few-shot example
    messages.append(current_query_message)
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
    processed_data = []
    for i in range(len(multi_turn_base_data)):
        entry = multi_turn_base_data[i]
        question_data = entry["question"]
        ground_truth = multi_turn_base_answer[i]["ground_truth"]
        initial_config = entry["initial_config"]

        # Assert number of turns matches between question and ground truth
        assert len(question_data) == len(ground_truth), (
            f"Mismatch in number of turns for entry {i}"
        )

        entry["num_total_turns_in_task"] = len(question_data) # Store original total turns
        entry["question"] = json.dumps(question_data)
        entry["initial_config"] = json.dumps(initial_config)
        entry["answer"] = json.dumps(ground_truth)
        processed_data.append(entry) # Keep processed entry

    multi_turn_base_data = processed_data # Use the processed list

    if curriculum_learning:
        # Create curriculum data with copies for each turn
        curriculum_data = []
        for entry in multi_turn_base_data:
            questions = json.loads(entry["question"])
            answers = json.loads(entry["answer"])
            total_turns = entry["num_total_turns_in_task"] # Use original total turns

            # Create copies for each turn number
            for j in range(1, total_turns + 1):
                curriculum_entry = copy.deepcopy(entry)
                # Keep only the first j turns of questions/answers
                curriculum_entry["question"] = json.dumps(copy.deepcopy(questions[:j]))
                curriculum_entry["answer"] = json.dumps(copy.deepcopy(answers[:j]))
                curriculum_entry["num_turns"] = j # Current turn number for this entry
                # Keep num_total_turns_in_task for potential filtering later
                curriculum_data.append(curriculum_entry)
        multi_turn_base_data = curriculum_data

    dataset = Dataset.from_list(multi_turn_base_data)
    dataset = dataset.map(
        lambda x: {
            "prompt": format_bfcl_prompt(
                involved_classes=x["involved_classes"],
                # User question is always the first turn's first message content
                user_question=json.loads(x["question"])[0][0]["content"],
                #include_few_shot=False, # Decide if few-shot needed per example
            ),
            # Ground truth bank now needs to correspond to the *current* turn number `j`
            # The 'answer' field already contains answers up to turn `j` if curriculum_learning=True
            "ground_truth_bank": copy.deepcopy(x["answer"]),
            "num_turns": x["num_turns"], # Current turn number for this curriculum step
            "num_total_turns_in_task": x["num_total_turns_in_task"], # Original total turns
            "id": x["id"],
            # Add other fields needed by the environment state initialization
            "involved_classes": x["involved_classes"],
            "initial_config": x["initial_config"],
        }
    )

    # Get unique IDs and split those first (using original IDs)
    unique_ids = sorted(list(set(dataset["id"])))
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.5, random_state=42)

    # Filter dataset based on IDs
    train_dataset = dataset.filter(lambda x: x["id"] in train_ids)
    test_dataset = dataset.filter(lambda x: x["id"] in test_ids)

    if curriculum_learning:
        # Sort both splits by num_turns while preserving randomization within same num_turns
        def sort_by_turns(split_dataset):
            df = split_dataset.to_pandas()
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
        sampling_args={ # Default stop tokens + added special tokens
            "stop": [
                "</tool>",
                "<TASK_FINISHED>",
                "<TASK_ERROR>",
                #"<|end_of_text|>", # Common EOS token
                #"<|im_end|>",      # Another common EOS/Stop token
                #"<|endoftext|>",   # Another common EOS token
                #"</s>",            # Llama EOS token
                #"<｜end of sentence｜>", # Add specific problematic token
                #"<｜begin of sentence｜>" # Add specific problematic token
            ],
            "include_stop_str_in_output": True,
        },
        mask_env_response: bool = True,
        max_steps_per_turn: int = 10, # Max tool interactions per assistant turn
        curriculum_learning: bool = True,
        use_latest_trl: bool = False,
        **kwargs,
    ):
        logger.info("Initializing BfclITEnv (Single Tool Call Flow)")
        self.tokenizer = None # will be set later
        self.env_mask = -100 # Default mask value

        self.max_num_turns = kwargs.pop("max_num_turns", 1) # Kept for dataset filtering compatibility

        super().__init__(
            few_shot=few_shot,
            mask_env_response=mask_env_response,
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
        self.dataset = None # Initialize dataset attribute
        self.eval_dataset = None # Initialize eval_dataset attribute


        self.max_steps_per_turn = max_steps_per_turn
        logger.info("Initializing Scoring Rubric")
        self.rubric = BfclITRubric()
        logger.info("Initializing LLM Parser")
        # Parser still extracts content between tags, call_tool handles the content format
        self.llm_parser = XMLParser(fields=["think", "tool"])
        self.use_latest_trl = use_latest_trl

    def get_dataset(self, max_num_turns: int = -1, **kwargs: Any) -> Dataset:
        if self.dataset is None:
            logger.info(f"Preprocessing dataset {self.dataset_name} for train split...")
            self.dataset = preprocess_bfcl_dataset(
                split="train",
                curriculum_learning=self.curriculum_learning,
            )
        # Filter based on current turn number if curriculum learning,
        # or total turns if not curriculum learning.
        filter_field = "num_turns" if self.curriculum_learning else "num_total_turns_in_task"
        effective_max_turns = max_num_turns if max_num_turns > 0 else self.max_num_turns

        if effective_max_turns > 0:
            logger.info(f"Filtering train dataset where {filter_field} <= {effective_max_turns}.")
            self.dataset = self.dataset.filter(
                lambda x: x[filter_field] <= effective_max_turns
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
            logger.info(f"Preprocessing dataset {self.dataset_name} for test split...")
            self.eval_dataset = preprocess_bfcl_dataset(
                split="test",
                curriculum_learning=self.curriculum_learning, # Use same setting as train
            )

        # Filter based on current turn number if curriculum learning,
        # or total turns if not curriculum learning.
        filter_field = "num_turns" if self.curriculum_learning else "num_total_turns_in_task"
        effective_max_turns = max_num_turns if max_num_turns > 0 else self.max_num_turns

        if effective_max_turns > 0:
             logger.info(f"Filtering eval dataset where {filter_field} <= {effective_max_turns}.")
             self.eval_dataset = self.eval_dataset.filter(
                 lambda x: x[filter_field] <= effective_max_turns
             )

        if max_turn_only:
            # If curriculum learning, keep max num_turns per id.
            # If not, this filter doesn't make sense unless applied before splitting.
            # Assuming curriculum learning for this flag's typical use.
            if self.curriculum_learning:
                logger.info("Filtering eval dataset to keep only the max turn entry per ID.")
                grouped = {}
                for item in self.eval_dataset:
                    item_id = item["id"]
                    if (
                        item_id not in grouped
                        or grouped[item_id]["num_turns"] < item["num_turns"]
                    ):
                        grouped[item_id] = item
                self.eval_dataset = Dataset.from_list(list(grouped.values()))
            else:
                logger.warning("max_turn_only=True ignored when curriculum_learning=False.")

        if n > 0:
            if len(self.eval_dataset) < n:
                 logger.warning(f"Requested {n} eval samples, but only {len(self.eval_dataset)} available after filtering. Using all available.")
                 n = len(self.eval_dataset)
            if n > 0: # Check again in case dataset became empty
                 self.eval_dataset = self.eval_dataset.shuffle(seed=42).select(range(n))
            else:
                 logger.warning("Eval dataset is empty after filtering.")
                 self.eval_dataset = None # Set to None if empty

        return self.eval_dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def _get_tool_interaction_count(
        self, messages: List[Dict[str, str]], debug: bool = False
    ) -> int:
        """Counts the number of tool interactions within the assistant message."""
        count = 0
        if messages and messages[-1].get("role") == "assistant":
            # Count occurrences of the result tag, as each call should produce one result
            count = messages[-1]["content"].count("</tool_result>")
        if debug:
            print(f"Tool Interaction Count: {count}")
        return count

    def is_completed(
        self, state: Dict[str, Any] = None, debug: bool = False, **kwargs: Any
    ) -> bool:
        """Checks if the current assistant turn is complete."""
        if state.get("completed", False):
            return True

        messages = state["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            return False

        assistant_response = messages[-1]["content"]
        tool_interactions = self._get_tool_interaction_count(messages, debug=debug)

        if "<TASK_FINISHED>" in assistant_response or "<TASK_ERROR>" in assistant_response:
            if debug: print(f"Found termination token. Entry Completed. Tool interactions: {tool_interactions}")
            return True

        if tool_interactions >= self.max_steps_per_turn:
            if debug: print(f"Reached max tool interactions ({self.max_steps_per_turn}). Entry Considered Completed.")
            # Logic in `step` will append TASK_ERROR if needed upon reaching max steps
            return True

        # Check for max completion length (handled in step, but good for robustness)
        sampling_params = kwargs.get("sampling_params")
        if sampling_params and hasattr(sampling_params, 'max_tokens') and sampling_params.max_tokens:
             if len(state.get("completion_ids", [])) >= sampling_params.max_tokens:
                 if debug: print(f"Reached max tokens ({sampling_params.max_tokens}). Entry Completed.")
                 return True

        return False

    current_entry_completed = is_completed # Alias

    def call_tool(
        self,
        tool_json_str: str, # Expecting a string containing a single JSON object
        state: Dict[str, Any] = None,
        debug: bool = False,
        ground_truth: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call a single tool based on JSON object string."""
        if ground_truth:
            # --- Ground Truth Execution (Modified for current turn's calls) ---
            if debug: print("Executing Ground Truth Tool Calls for current turn")
            try:
                # 'answer' in dataset_row is JSON string of list of lists (turns -> calls)
                all_turns_answers = json.loads(state["dataset_row"]["answer"])
                current_turn_index = state["dataset_row"]["num_turns"] - 1 # 0-indexed

                # Ensure the current turn exists in the ground truth
                if current_turn_index >= len(all_turns_answers):
                    # This might happen if generation goes beyond GT turns, treat as no GT calls needed
                    logger.warning(f"State {state.get('id', 'N/A')}: Current turn {current_turn_index+1} exceeds ground truth turns ({len(all_turns_answers)}). No GT calls to execute.")
                    return "Ground truth execution skipped (turn limit reached).", state

                tool_calls_for_current_turn = all_turns_answers[current_turn_index]

                if not isinstance(tool_calls_for_current_turn, list):
                    print(f"Error: Expected list of tool calls in ground truth for turn {current_turn_index+1}, got: {tool_calls_for_current_turn}")
                    raise Exception("Error in ground truth tool execution format!")

                # Initialize GT environment if missing (should be done in _initialize_environments)
                if "ground_truth_environment" not in state or not state["ground_truth_environment"]:
                    print("Error: Ground truth environment not initialized.")
                    raise Exception("Ground truth environment missing.")

                # Map method names to their class instances in the GT environment
                method_to_instance = {}
                for class_name, instance in state["ground_truth_environment"].items():
                    for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
                        if not method_name.startswith("_"):
                            method_to_instance[method_name] = class_name

                # Execute each function call string for the *current* turn
                state["ground_truth_successful_calls"] = [] # Track successful GT calls for this turn
                for func_call_str in tool_calls_for_current_turn:
                    if not isinstance(func_call_str, str) or "(" not in func_call_str:
                        print(f"Error: Invalid ground truth function call format: {func_call_str}")
                        raise Exception("Error in ground truth tool execution format!")

                    method_name = func_call_str.split("(")[0].strip()
                    if method_name not in method_to_instance:
                        print(f"Error: Method '{method_name}' from ground truth not found in environment instances.")
                        print(f"Available methods: {list(method_to_instance.keys())}")
                        raise Exception("Error finding ground truth method!")

                    class_name = method_to_instance[method_name]
                    # IMPORTANT: Use the state dict directly in eval for safety
                    modified_call = f"state['ground_truth_environment']['{class_name}'].{func_call_str}"

                    if debug: print(f"Executing ground truth call: {func_call_str}")
                    try:
                        result = eval(modified_call, {"state": state}) # Pass state explicitly
                        # Store the call string itself on success for potential comparison/reward
                        state["ground_truth_successful_calls"].append(func_call_str)
                    except Exception as e:
                        print(f"Error executing ground truth call '{modified_call}': {e}")
                        # Raise to signal GT failure, crucial for reward calculation
                        raise Exception(f"Error during ground truth tool execution: {e}")

                return "Ground truth execution successful for turn.", state
            except Exception as e:
                print(f"Ground Truth Execution Failed: {e}")
                print(f"State ID: {state.get('id', 'N/A')}")
                print(f"Ground Truth Answer Slice: {state.get('dataset_row', {}).get('answer', 'N/A')}")
                raise Exception(f"Error in ground truth tool execution is not expected!! Error: {e}")

        # --- Handling Model Tool Call (Single JSON Object) ---
        tool_call_result_str = ""
        try:
            # Ensure tool_json_str is not None or empty before parsing
            if not tool_json_str or not tool_json_str.strip():
                 if debug:logger.warning(f"State {state.get('id', 'N/A')}: Received empty tool call string.")
                 return json.dumps("Error: Empty tool command received."), state

            # Attempt to parse the string as a single JSON object
            tool_call = json.loads(tool_json_str)

            if not (
                isinstance(tool_call, dict)
                and "name" in tool_call
                and "args" in tool_call
                and isinstance(tool_call["args"], dict)
            ):
                tool_call_result_str = f"Function Call {tool_json_str} Failed. Error: Invalid format (must be JSON object with 'name' and 'args' dict)."
                return json.dumps(tool_call_result_str), state # Return error in list format for consistency

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Convert lists to tuples if needed by specific tools (remains useful)
            for key, value in tool_args.items():
                if isinstance(value, list):
                    tool_args[key] = tuple(value)

            if debug: print(f"Attempting Tool Call: {tool_name}({tool_args})")

            found_method = False
            target_instance = None
            if "environment" not in state or not state["environment"]:
                tool_call_result_str = f"Function Call {tool_call} Failed. Error: Environment not initialized."
                return json.dumps(tool_call_result_str), state

            # Find the method in the initialized environment instances
            for class_instance in state["environment"].values():
                if hasattr(class_instance, tool_name):
                    found_method = True
                    target_instance = class_instance
                    if debug: print(f"Found method {tool_name} in class {target_instance.__class__.__name__}")
                    tool_func = getattr(target_instance, tool_name)
                    break

            if not found_method:
                available_tools_msg = self._get_available_tools_message(state)
                tool_call_result_str = f"Function Call {tool_call} Failed. Error: Method '{tool_name}' not found. {available_tools_msg}"
                return json.dumps([tool_call_result_str]), state

            # Execute the tool function
            try:
                result = tool_func(**tool_args)
                result_str = str(result) if result is not None else "Success" # Handle None results

                # Check for explicit errors returned by the tool itself
                # Be specific about error checking if possible
                if isinstance(result, dict) and 'error' in result:
                    error_detail = result.get('error', 'Unknown tool error')
                    tool_call_result_str = f"Function Call {tool_call} Failed during execution. Error: {error_detail}"
                elif isinstance(result, str) and "error" in result.lower() and len(result) < 150: # Heuristic
                    tool_call_result_str = f"Function Call {tool_call} Failed during execution. Error: {result}"
                else:
                    tool_call_result_str = f"Function Call {tool_call} Succeeded. Result: {result_str}"
                    # Log successful call (append the dict)
                    if "successful_func_calls" not in state: state["successful_func_calls"] = []
                    state["successful_func_calls"].append(tool_call) # Append the successful call details

            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                error_msg = f"Function Call {tool_call} Failed during execution. Error: {e}. Traceback: {tb_str}"
                logger.error(error_msg) # Log the full error
                tool_call_result_str = f"Function Call {tool_call} Failed during execution. Error: {e}"

            # Return result string within a list for consistency with multi-call format expectation downstream (if any)
            return json.dumps(tool_call_result_str), state

        except json.JSONDecodeError as e:
            if debug: logger.warning(f"State {state.get('id', 'N/A')}: JSONDecodeError parsing tool call: '{tool_json_str}'. Error: {e}")
            return json.dumps(f"Error decoding tool call JSON: {e}. Ensure the format is a single JSON object."), state
        except Exception as e:
            logger.exception(f"State {state.get('id', 'N/A')}: Unexpected error in call_tool for call '{tool_json_str}': {e}")
            return json.dumps(f"Unexpected error processing tool call: {e}"), state

    def _get_available_tools_message(self, state: Dict[str, Any]) -> str:
        """Helper to create a message listing available tools."""
        # Use the pre-computed list from format_bfcl_prompt if available
        # Otherwise, compute it (e.g., if called standalone)
        if "dataset_row" in state and "involved_classes" in state["dataset_row"]:
             tool_names_str = _get_tool_names_from_involved_classes(state["dataset_row"]["involved_classes"])
             return f"Available Tools: {tool_names_str}"
        else:
             logger.warning("Could not determine available tools: 'involved_classes' missing in state.")
             return "Could not determine available tools."


    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        """Not used in this single-assistant-turn flow."""
        raise NotImplementedError("env_response is not used in BfclITEnv.")

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
            # Double check completion status
            if state.get("completed", False):
                continue

            current_messages = state["messages"]
            if not current_messages: # Should not happen if initialized correctly
                 logger.error(f"State {i}: Empty messages list found. Marking as error.")
                 state["completed"] = True
                 state["error_message"] = "Internal error: Empty messages list."
                 continue

            # Check the role of the actual last message to decide the call type
            if current_messages[-1]["role"] == "user":
                start_states_data.append({"index": i, "messages": current_messages})
            elif current_messages[-1]["role"] == "assistant":
                 if debug:
                    logger.debug(f"State {i} checking for continuation. Last message ends with: '{current_messages[-1]['content'][-50:]}'")
                 # Check if the last assistant message ended with a tool result, requiring continuation
                 if current_messages[-1]["content"].strip().endswith("</tool_result> Ok,") or current_messages[-1]["content"].strip().endswith("</tool_result> Wait,"):
                      continue_states_data.append({"index": i, "messages": current_messages})
                 else:
                      # If assistant message doesn't end with result, it might be finished or errored.
                      # is_completed check should handle this, but add a warning.
                      if debug: logger.warning(f"State {i}: Last message is assistant but doesn't end with </tool_result>. Checking completion status.")
                      if self.is_completed(state, debug=debug, sampling_params=sampling_params):
                           state["completed"] = True # Mark completed based on is_completed logic
                           if debug: print(f"State {i} marked completed by is_completed check.")
                      else:
                           # This case is ambiguous - why continue if not ending in result and not completed?
                           if debug: logger.error(f"State {i}: Ambiguous state - assistant message without result, but not completed. Marking error.")
                           state["completed"] = True
                           state["error_message"] = "Internal error: Ambiguous assistant state."

            else: # e.g., tool_result role if it were used
                logger.error(f"State {i}: Unexpected last message role '{current_messages[-1]['role']}'. Marking as error.")
                state["completed"] = True
                state["error_message"] = f"Internal error: Invalid last message role '{current_messages[-1]['role']}'."


        # --- Perform LLM Calls Separately ---
        all_llm_responses = {} # Store responses mapped by original index: { index: llm_response_obj }

        # Call for states starting a new message (after user prompt)
        if start_states_data:
            start_indices = [d["index"] for d in start_states_data]
            start_messages = [d["messages"] for d in start_states_data]
            if debug: print(f"---> Calling LLM for {len(start_indices)} states (add_generation_prompt=True)")
            try:
                # Ensure tokenizer is set before calling LLM
                if self.tokenizer is None and hasattr(llm, 'tokenizer'):
                     self.tokenizer = llm.tokenizer
                     logger.info("Tokenizer set from LLM object.")
                elif self.tokenizer is None:
                     logger.error("Tokenizer not set in environment or LLM object!")
                     raise ValueError("Tokenizer is required for processing LLM responses but is not set.")

                start_responses = llm.chat(
                    start_messages, sampling_params=sampling_params, use_tqdm=False,
                    add_generation_prompt=True,
                    continue_final_message=False, # Start new assistant message
                )
                if len(start_responses) != len(start_indices):
                    logger.error(f"LLM call (start group) mismatch: Sent {len(start_indices)} prompts, received {len(start_responses)} responses.")
                    for original_index in start_indices:
                        if original_index not in all_llm_responses:
                            states[original_index]["completed"] = True
                            states[original_index]["error_message"] = "LLM response count mismatch (start group)."
                else:
                    for idx, response_obj in enumerate(start_responses):
                        original_index = start_indices[idx]
                        all_llm_responses[original_index] = response_obj
            except Exception as e:
                logger.exception(f"Error during LLM chat call (start group): {e}")
                for original_index in start_indices:
                    states[original_index]["completed"] = True
                    states[original_index]["error_message"] = f"LLM chat call failed (start group): {e}"

        # Call for states continuing a message (after tool_result)
        if continue_states_data:
            continue_indices = [d["index"] for d in continue_states_data]
            continue_messages = [d["messages"] for d in continue_states_data]
            if debug:
                print(f"---> Calling LLM for {len(continue_indices)} states (continue_final_message=True)")
            try:
                 # Ensure tokenizer is set
                if self.tokenizer is None and hasattr(llm, 'tokenizer'):
                     self.tokenizer = llm.tokenizer
                     logger.info("Tokenizer set from LLM object.")
                elif self.tokenizer is None:
                     logger.error("Tokenizer not set in environment or LLM object!")
                     raise ValueError("Tokenizer is required for processing LLM responses but is not set.")

                continue_responses = llm.chat(
                    continue_messages, sampling_params=sampling_params, use_tqdm=False,
                    add_generation_prompt=False, # Don't add prompt again
                    continue_final_message=True, # Append to existing assistant message
                )
                if debug:
                    logger.debug(f"Raw responses from continuation call (Count: {len(continue_responses)}):")
                    for idx, resp_obj in enumerate(continue_responses):
                        original_index = continue_indices[idx]
                        logger.debug(f"--- Response for State {original_index} ---")
                        # Log relevant parts of the RequestOutput object
                        logger.debug(f"  Prompt Tokens: {len(resp_obj.prompt_token_ids) if resp_obj.prompt_token_ids else 'N/A'}")
                        if resp_obj.outputs:
                            logger.debug(f"  Output Text: '{resp_obj.outputs[0].text}'")
                            logger.debug(f"  Output Tokens: {resp_obj.outputs[0].token_ids}")
                            logger.debug(f"  Finish Reason: {resp_obj.outputs[0].finish_reason}")
                        else:
                            logger.debug("  No outputs in response object.")
                        logger.debug(f"  Error: {resp_obj.error if hasattr(resp_obj, 'error') else 'N/A'}") # Check if vLLM adds error info
                        logger.debug(f"-------------------------")
                if len(continue_responses) != len(continue_indices):
                    logger.error(f"LLM call (continue group) mismatch: Sent {len(continue_indices)} prompts, received {len(continue_responses)} responses.")
                    for original_index in continue_indices:
                        if original_index not in all_llm_responses:
                            states[original_index]["completed"] = True
                            states[original_index]["error_message"] = "LLM response count mismatch (continue group)."
                else:
                    for idx, response_obj in enumerate(continue_responses):
                        original_index = continue_indices[idx]
                        if original_index in all_llm_responses:
                            logger.warning(f"State {original_index} received response from both start and continue groups. Overwriting with continue response.")
                        all_llm_responses[original_index] = response_obj
            except Exception as e:
                if debug: logger.exception(f"Error during LLM chat call (continue group): {e}")
                for original_index in continue_indices:
                    if not states[original_index].get("completed", False):
                        states[original_index]["completed"] = True
                        states[original_index]["error_message"] = f"LLM chat call failed (continue group): {e}"

        # --- Process All Received LLM Responses ---
        if not all_llm_responses:
            if debug: print("No LLM responses received or processed in this step.")
            # Check if any states are still live but didn't get a response (shouldn't happen)
            for i in live_indices:
                 if i not in all_llm_responses and not states[i].get("completed", False):
                      if debug: logger.error(f"State {i} was live but received no LLM response. Marking error.")
                      states[i]["completed"] = True
                      states[i]["error_message"] = "Internal error: No LLM response received."
            return states # Return states as they might have been updated with errors

        if debug: print(f"--- Processing {len(all_llm_responses)} LLM responses ---")

        processed_indices = sorted(all_llm_responses.keys())
        for live_idx in processed_indices:
            state = states[live_idx]
            llm_response_obj = all_llm_responses[live_idx]

            # Skip processing if the state was marked completed due to an LLM error above
            if state.get("completed", False) and "LLM" in state.get("error_message", ""):
                continue
            if state.get("completed", False): # General check
                logger.warning(f"State {live_idx} was already completed before processing its LLM response. Skipping.")
                continue

            # Ensure tokenizer is available
            if self.tokenizer is None:
                 logger.error(f"State {live_idx}: Tokenizer is None during response processing. Cannot proceed.")
                 state["completed"] = True
                 state["error_message"] = "Internal error: Tokenizer not available."
                 continue

            new_response_text = llm_response_obj.outputs[0].text
            new_response_ids = llm_response_obj.outputs[0].token_ids

            if debug and live_idx == processed_indices[0]: # Log for the first state processed in this batch
                print(f"State {live_idx} LLM Output Segment: '{new_response_text}'")

            # Initialize state fields if needed
            if "completion_ids" not in state: state["completion_ids"] = []
            if "completion_mask" not in state: state["completion_mask"] = []
            if "prompt_ids" not in state: state["prompt_ids"] = []
            if "successful_func_calls" not in state: state["successful_func_calls"] = []
            if not state["prompt_ids"] and llm_response_obj.prompt_token_ids:
                state["prompt_ids"] = list(llm_response_obj.prompt_token_ids)

            # Append new text segment
            last_message_role_before_call = None
            if live_idx in [d['index'] for d in start_states_data]:
                last_message_role_before_call = 'user'
            elif live_idx in [d['index'] for d in continue_states_data]:
                last_message_role_before_call = 'assistant'

            if last_message_role_before_call == 'user':
                # Start a new assistant message
                state["messages"].append({"role": "assistant", "content": new_response_text})
            elif last_message_role_before_call == 'assistant':
                # Append to the last assistant message
                if state["messages"] and state["messages"][-1]["role"] == "assistant":
                     state["messages"][-1]["content"] += new_response_text
                else:
                     # Should not happen if logic is correct, but handle defensively
                     if debug:logger.error(f"State {live_idx}: Tried to continue assistant message, but last message role was not assistant. Starting new.")
                     state["messages"].append({"role": "assistant", "content": new_response_text})
            else:
                if debug: logger.error(f"State {live_idx}: Could not determine if starting or continuing. Last role before call unknown. Marking error.")
                state["completed"] = True
                state["error_message"] = "Internal error: Ambiguous state for appending LLM response."
                continue

            # Update completion IDs/mask for the *generated* part
            state["completion_ids"].extend(new_response_ids)
            # Mask for generated tokens is initially 1 (learnable)
            state["completion_mask"].extend([1] * len(new_response_ids))

            # --- Check for Stop Conditions and Process Tool Calls ---
            current_content = state["messages"][-1]["content"] # Full accumulated content for this assistant turn
            completed_this_step = False
            needs_recalculation = False # Flag if content/tokens changed requiring mask rebuild

            # 1. Check for TERMINATION tokens
            termination_token = None
            if "<TASK_FINISHED>" in current_content: termination_token = "<TASK_FINISHED>"
            elif "<TASK_ERROR>" in current_content: termination_token = "<TASK_ERROR>"

            if termination_token:
                if debug and live_idx == processed_indices[0]: print(f"State {live_idx}: Found termination token '{termination_token}'.")
                completed_this_step = True
                state["completed"] = True
                # Truncate content *after* the first termination token
                first_term_idx = current_content.find(termination_token)
                target_content_len = first_term_idx + len(termination_token)
                if len(current_content) > target_content_len:
                    if debug and live_idx == processed_indices[0]: print(f"State {live_idx}: Truncating content after first '{termination_token}'.")
                    state["messages"][-1]["content"] = current_content[:target_content_len]
                    needs_recalculation = True # Tokens need recalculation due to truncation

            # 2. Check for Tool Call Stop Token (only if not already terminated)
            # Use rstrip to handle potential trailing whitespace before stop token
            elif current_content.rstrip().endswith("</tool>"):
                stop_token_found = "</tool>"
                if debug and live_idx == processed_indices[0]: print(f"State {live_idx}: Detected </tool>")
                try:
                    # Extract content between the last <tool> and </tool>
                    last_tool_start_idx = current_content.rfind("<tool>")
                    if last_tool_start_idx != -1:
                        # Get the content between the tags, excluding the tags themselves
                        tool_content_start = last_tool_start_idx + len("<tool>")
                        tool_content_end = current_content.rfind("</tool>") # Find the *last* closing tag relative to content end
                        if tool_content_end > tool_content_start:
                             tool_json_str = current_content[tool_content_start:tool_content_end].strip()

                             # Call the tool with the extracted JSON string
                             tool_result_json, state = self.call_tool(tool_json_str, state=state, debug=debug)
                             if "error" in tool_result_json.lower():
                                 tool_result_str = f"<tool_result>{tool_result_json}</tool_result> Wait, "
                             else:
                                 tool_result_str = f"<tool_result>{tool_result_json}</tool_result> Ok, " # Note: Removed space inside tags


                             # Append result to message content
                             state["messages"][-1]["content"] += tool_result_str

                             # Tokenize and update completion_ids and mask for the result part
                             # Use add_special_tokens=False for environment-injected content
                             result_tokens = self.tokenizer.encode(tool_result_str, add_special_tokens=False)
                             state["completion_ids"].extend(result_tokens)
                             state["completion_mask"].extend([self.env_mask] * len(result_tokens)) # Mask the result
                             needs_recalculation = False # Tokens added, but no truncation needed here

                             # Check if max tool interactions reached AFTER this call
                             tool_interactions = self._get_tool_interaction_count(state["messages"])
                             if tool_interactions >= self.max_steps_per_turn:
                                 if debug:logger.warning(f"State {live_idx}: Reached max tool interactions ({self.max_steps_per_turn}) after tool call.")
                                 error_marker = "\n<TASK_ERROR>Max tool interactions reached.</TASK_ERROR>"
                                 state["messages"][-1]["content"] += error_marker
                                 # Tokenize and add error marker
                                 error_tokens = self.tokenizer.encode(error_marker, add_special_tokens=False)
                                 state["completion_ids"].extend(error_tokens)
                                 state["completion_mask"].extend([1] * len(error_tokens)) # Error message is learnable
                                 completed_this_step = True; state["completed"] = True
                                 needs_recalculation = False # Added tokens, no truncation
                             else:
                                 # Ready for next LLM call (continue)
                                 completed_this_step = False; state["completed"] = False

                        else: # Malformed <tool>...</tool> pair
                            if debug: logger.warning(f"State {live_idx}: Found </tool> but couldn't extract valid content after last <tool>.")
                            error_str = "<tool_result>\"Error: Malformed tool call structure.\"</tool_result> Wait, "
                            state["messages"][-1]["content"] += error_str
                            error_tokens = self.tokenizer.encode(error_str, add_special_tokens=False)
                            state["completion_ids"].extend(error_tokens)
                            state["completion_mask"].extend([self.env_mask] * len(error_tokens))
                            completed_this_step = False; state["completed"] = False # Continue, let model potentially fix
                            needs_recalculation = False
                    else: # Found </tool> without preceding <tool> in the current assistant message segment
                        if debug: logger.warning(f"State {live_idx}: Found </tool> without preceding <tool> tag.")
                        error_str = "<tool_result>\"Error: Stray </tool> tag found.\"</tool_result> Wait, "
                        state["messages"][-1]["content"] += error_str
                        error_tokens = self.tokenizer.encode(error_str, add_special_tokens=False)
                        state["completion_ids"].extend(error_tokens)
                        state["completion_mask"].extend([self.env_mask] * len(error_tokens))
                        completed_this_step = False; state["completed"] = False # Continue
                        needs_recalculation = False

                except Exception as e:
                    logger.exception(f"State {live_idx}: Error during tool call processing: {e}")
                    error_str = f"<tool_result>\"Error: Exception during tool processing: {e}\"</tool_result> Wait, "
                    state["messages"][-1]["content"] += error_str
                    error_tokens = self.tokenizer.encode(error_str, add_special_tokens=False)
                    state["completion_ids"].extend(error_tokens)
                    state["completion_mask"].extend([self.env_mask] * len(error_tokens))
                    # Add TASK_ERROR as this is likely unrecoverable by the model
                    error_marker = "\n<TASK_ERROR>Internal error during tool execution.</TASK_ERROR>"
                    state["messages"][-1]["content"] += error_marker
                    error_tokens_term = self.tokenizer.encode(error_marker, add_special_tokens=False)
                    state["completion_ids"].extend(error_tokens_term)
                    state["completion_mask"].extend([1] * len(error_tokens_term)) # Learnable error message
                    completed_this_step = True; state["completed"] = True
                    needs_recalculation = False

            # 3. Check max tokens (only if not already completed)
            current_completion_len = len(state.get("completion_ids", []))
            max_tok = getattr(sampling_params, 'max_tokens', None) # Safe access
            if not completed_this_step and max_tok is not None and current_completion_len >= max_tok:
                if debug:logger.warning(f"State {live_idx}: Reached max tokens ({max_tok}). Current: {current_completion_len}. Truncating.")
                # Truncate completion tokens and mask
                state["completion_ids"] = state["completion_ids"][:max_tok]
                state["completion_mask"] = state["completion_mask"][:max_tok]
                needs_recalculation = True # Content needs recalculation based on truncated tokens
                completed_this_step = True
                state["completed"] = True
                # Add TASK_ERROR marker textually if not already present
                if "<TASK_FINISHED>" not in state["messages"][-1]["content"] and \
                   "<TASK_ERROR>" not in state["messages"][-1]["content"]:
                    state["messages"][-1]["content"] += "\n<TASK_ERROR>Max tokens reached.</TASK_ERROR>"
                    # Note: The actual tokens for this added text might exceed max_tokens now,
                    # but we prioritize having the termination marker. Recalculation handles this.

            # --- Recalculate Tokens/Mask if Content was Truncated ---
            if needs_recalculation:
                if debug and live_idx == processed_indices[0]: print(f"State {live_idx}: Recalculating tokens and mask due to content change.")
                try:
                    # Get the final intended content string
                    final_content_str = state["messages"][-1]["content"]
                    # Re-tokenize the final content string
                    # Important: Use add_special_tokens=False for assistant content part
                    final_content_tokens = self.tokenizer.encode(final_content_str, add_special_tokens=False)

                    # Assume prompt_ids are correct and stable
                    prompt_len = len(state.get("prompt_ids", []))

                    # The new completion_ids are the part after the prompt
                    # This assumes the prompt structure was handled correctly by llm.chat
                    # A more robust way might involve tokenizing the full history, but let's try this first.
                    # If llm.chat prepends prompt tokens, this might be incorrect.
                    # Let's assume completion_ids should be based on the final_content_str directly for now.
                    state["completion_ids"] = final_content_tokens

                    # Rebuild the mask based on the *final* content string
                    mask = []
                    # Split by tool result tags to identify masked sections
                    # Use the final content string for splitting
                    segments = final_content_str.split("<tool_result>")
                    first_segment = True
                    processed_tokens_count = 0
                    for segment in segments:
                        if not first_segment and "</tool_result>" in segment:
                            # Part between <tool_result> and </tool_result> should be masked
                            result_part, rest_part = segment.split("</tool_result>", 1)
                            # Tokenize the parts including the tags
                            result_tokens_segment = self.tokenizer.encode("<tool_result>" + result_part + "</tool_result>", add_special_tokens=False)
                            mask.extend([self.env_mask] * len(result_tokens_segment))
                            processed_tokens_count += len(result_tokens_segment)
                            # Tokenize the part after </tool_result>
                            rest_tokens_segment = self.tokenizer.encode(rest_part, add_special_tokens=False)
                            mask.extend([1] * len(rest_tokens_segment)) # Learnable
                            processed_tokens_count += len(rest_tokens_segment)
                        else:
                            # First segment or segment without a result tag is learnable
                            part_tokens = self.tokenizer.encode(segment, add_special_tokens=False)
                            if part_tokens: # Avoid adding empty masks
                                 mask.extend([1] * len(part_tokens))
                                 processed_tokens_count += len(part_tokens)
                        first_segment = False

                    # Ensure mask length matches the final token length
                    state["completion_mask"] = mask[:len(state["completion_ids"])]
                    if len(state["completion_mask"]) != len(state["completion_ids"]):
                        logger.warning(f"State {live_idx}: Mask length mismatch after rebuild ({len(state['completion_mask'])} vs {len(state['completion_ids'])}). Adjusting.")
                        min_len_rebuild = min(len(state["completion_ids"]), len(state["completion_mask"]))
                        state["completion_ids"] = state["completion_ids"][:min_len_rebuild]
                        state["completion_mask"] = state["completion_mask"][:min_len_rebuild]

                except Exception as e_recalc:
                    logger.exception(f"State {live_idx}: Error recalculating tokens/mask after content change: {e_recalc}. State might be inconsistent.")
                    # Mark as error if recalculation fails
                    state["completed"] = True
                    state["error_message"] = f"Internal error: Failed to recalculate tokens/mask: {e_recalc}"

        if debug: print("-" * 30 + f" Step End (Processed {len(all_llm_responses)} LLM responses) " + "-" * 30)
        return states


    def _initialize_environments(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Initialize environment instances for the given states"""
        # Ensure tokenizer is available if needed within class init
        if self.tokenizer is None and any('tokenizer' in inspect.signature(cls.__init__).parameters for cls in [getattr(importlib.import_module(self.CLASS_FILE_PATH_MAPPING[cn]), cn) for cn in self.CLASS_FILE_PATH_MAPPING]):
             logger.warning("Some environment classes might require tokenizer during init, but it's not set yet.")
             # Attempt to get it from the first state's potential LLM object if passed? Risky.
             # Best practice: Ensure tokenizer is set in __init__ or before calling generate.

        for i in range(len(states)):
            # Use a unique ID based on the state object itself for this batch run
            instance_id = id(states[i])
            states[i]["instance_id"] = instance_id # Store for reference

            # Initialize only if not already present for this instance_id in this run
            if instance_id not in self.env_instances:
                self.env_instances[instance_id] = {"main": {}, "ground_truth": {}, "initial_instance": {}}

                involved_classes = states[i]["dataset_row"]["involved_classes"]
                initial_config_str = states[i]["dataset_row"].get("initial_config", "{}")
                try:
                    initial_config = json.loads(initial_config_str)
                except json.JSONDecodeError:
                    logger.error(f"State {i} (ID: {states[i].get('id', 'N/A')}): Invalid JSON in initial_config: {initial_config_str}")
                    initial_config = {} # Use empty config on error

                for class_name in involved_classes:
                    try:
                        module_name = self.CLASS_FILE_PATH_MAPPING[class_name]
                        module = importlib.import_module(module_name)
                        class_ = getattr(module, class_name)

                        # Create instances
                        # Pass tokenizer if the class __init__ accepts it
                        init_params = inspect.signature(class_.__init__).parameters
                        kwargs = {}
                        if 'tokenizer' in init_params:
                             kwargs['tokenizer'] = self.tokenizer

                        class_instance = class_(**kwargs)
                        ground_truth_class_instance = class_(**kwargs)
                        initial_instance_copy = class_(**kwargs)


                        # Configure non-stateless classes using _load_scenario if it exists
                        if class_name not in self.STATELESS_CLASSES and hasattr(class_instance, '_load_scenario'):
                            class_initial_config = initial_config.get(class_name, {})
                            # Use deepcopy to avoid shared state issues across instances
                            class_instance._load_scenario(copy.deepcopy(class_initial_config))
                            ground_truth_class_instance._load_scenario(copy.deepcopy(class_initial_config))
                            initial_instance_copy._load_scenario(copy.deepcopy(class_initial_config))
                        elif class_name not in self.STATELESS_CLASSES and not hasattr(class_instance, '_load_scenario'):
                             logger.warning(f"Class {class_name} is stateful but lacks a _load_scenario method for initialization.")


                        # Store instances in the central dict for this instance_id
                        self.env_instances[instance_id]["main"][class_name] = class_instance
                        self.env_instances[instance_id]["ground_truth"][class_name] = ground_truth_class_instance
                        self.env_instances[instance_id]["initial_instance"][class_name] = initial_instance_copy

                    except KeyError:
                        logger.error(f"State {i} (ID: {states[i].get('id', 'N/A')}): Class '{class_name}' not found in CLASS_FILE_PATH_MAPPING.")
                        # Mark state as errored?
                        states[i]["completed"] = True
                        states[i]["error_message"] = f"Initialization failed: Unknown class {class_name}"
                    except ImportError:
                        logger.error(f"State {i} (ID: {states[i].get('id', 'N/A')}): Could not import module for class '{class_name}'.")
                        states[i]["completed"] = True
                        states[i]["error_message"] = f"Initialization failed: Cannot import module for {class_name}"
                    except AttributeError:
                        logger.error(f"State {i} (ID: {states[i].get('id', 'N/A')}): Could not find class '{class_name}' in its module.")
                        states[i]["completed"] = True
                        states[i]["error_message"] = f"Initialization failed: Cannot find class {class_name} in module"
                    except Exception as e:
                        logger.exception(f"State {i} (ID: {states[i].get('id', 'N/A')}): Error initializing environment for class '{class_name}': {e}")
                        states[i]["completed"] = True
                        states[i]["error_message"] = f"Initialization failed for {class_name}: {e}"


            # Always link the state dict to the centrally managed instances
            states[i]["environment"] = self.env_instances[instance_id]["main"]
            states[i]["ground_truth_environment"] = self.env_instances[instance_id]["ground_truth"]
            states[i]["initial_environment"] = self.env_instances[instance_id]["initial_instance"]

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
        tokenizer: Any = None, # Allow passing tokenizer explicitly
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:

        # --- Tokenizer Setup ---
        if tokenizer is not None:
            self.tokenizer = tokenizer
            logger.info("Tokenizer explicitly provided to generate method.")
        elif hasattr(llm, 'tokenizer') and llm.tokenizer is not None:
            self.tokenizer = llm.tokenizer
            logger.info("Tokenizer obtained from LLM object.")
        elif self.tokenizer is None:
             # If still None, try to load default or raise error
             logger.error("Tokenizer not provided and not found on LLM object. Cannot proceed.")
             raise ValueError("Tokenizer is required for BfclITEnv but was not provided.")

        # --- Sampling Params Setup ---
        custom_sp = sampling_params.clone()
        # Apply env-specific sampling args (stop tokens etc.)
        # Make sure stop tokens are unique and handle None case
        env_stop = self.sampling_args.get("stop", [])
        original_stop = getattr(custom_sp, "stop", []) or []
        combined_stop = sorted(list(set(env_stop + original_stop)))
        setattr(custom_sp, "stop", combined_stop)

        # Apply other env-specific args
        for k, v in self.sampling_args.items():
            if k != "stop": # Avoid overwriting combined stop list
                setattr(custom_sp, k, v)
        logger.info(f"Using sampling params: stop={custom_sp.stop}, include_stop={getattr(custom_sp, 'include_stop_str_in_output', 'Not Set')}, max_tokens={getattr(custom_sp, 'max_tokens', 'Not Set')}")


        if dataset_rows is None or len(prompts) != len(dataset_rows):
            raise ValueError("dataset_rows must be provided and match length of prompts")

        # --- State Initialization ---
        states = []
        for i, m in enumerate(prompts):
            # Ensure dataset_row has necessary fields from preprocessing
            row_data = copy.deepcopy(dataset_rows[i])
            required_fields = ["involved_classes", "initial_config", "answer", "num_turns", "id"]
            if not all(field in row_data for field in required_fields):
                 logger.error(f"Dataset row {i} (ID: {row_data.get('id', 'N/A')}) is missing required fields. Fields: {list(row_data.keys())}")
                 # Skip this row or handle error appropriately
                 continue # Simple skip for now

            initial_state = {
                "messages": copy.deepcopy(m), # Starts with user message
                "prompt_messages_count": len(m), # Should be 1 (user message) or more if few-shot used
                "prompt_ids": [], # Will be populated by LLM response object
                "completed": False,
                "completion_ids": [],
                "completion_mask": [],
                "dataset_row": row_data, # Use the validated row data
                "successful_func_calls": [], # List for successful model calls
                "ground_truth_successful_calls": [], # List for successful GT calls
                "id": row_data.get("id", f"unidentified_{i}"), # Add ID for logging
                "error_message": None, # For storing errors
                "ground_truth_error": None, # For storing GT execution errors
            }
            states.append(initial_state)

        if not states:
             logger.error("No valid states initialized, possibly due to missing fields in all dataset rows.")
             # Return empty structure matching expected output format
             return { "ids": [], "messages": [], "mask": [], "states": [] }


        if debug:
            print(f"Initial User Prompt (State 0): {states[0]['messages'][0]['content'][:500]}...") # Print start of prompt
            print(f"Number of Rollouts: {len(states)}")

        # --- Initialize Environments ---
        try:
            states = self._initialize_environments(states)
            # Check if any states failed initialization
            if any((s.get("error_message") or "").startswith("Initialization failed") for s in states):
                 logger.error("Some environment initializations failed. See previous logs.")
                 # Proceed with potentially fewer active states, or handle as fatal error?
                 # For now, proceed, step loop will skip completed/errored states.
        except Exception as e:
            logger.exception(f"Fatal error during environment initialization: {e}")
            # Mark all states as errored
            for s in states:
                if not s.get("completed", False): # Avoid overwriting specific init errors
                     s["completed"] = True
                     s["error_message"] = f"Fatal environment initialization failed: {e}"
            # Cleanup and return immediately
            self.cleanup_instances()
            return self._package_generate_output(states)


        # --- Main Generation Loop ---
        step_count = 0
        # Adjust max_overall_steps based on max_tokens and typical interactions
        # Example: Allow roughly double the tool interactions plus some buffer
        max_overall_steps = (self.max_steps_per_turn * 2) + 5
        all_completed = all(s.get("completed", False) for s in states) # Check initial state

        while not all_completed and step_count < max_overall_steps:
            step_count += 1
            if debug: print(f"\n--- Starting Generation Step {step_count} ---")
            try:
                states = self.step(states, llm, custom_sp, debug=debug)
            except Exception as e:
                logger.exception(f"Fatal error during generation step {step_count}: {e}")
                # Mark all *currently live* states as errored
                for s in states:
                    if not s.get("completed", False):
                        s["completed"] = True
                        s["error_message"] = f"Generation failed at step {step_count}: {e}"
                break # Exit loop on fatal error
            if debug:
                logger.debug("--- End of Step Status ---")
                for idx, s in enumerate(states):
                    logger.debug(f"State {idx} (ID: {s.get('id', 'N/A')}): Completed={s.get('completed', False)}, Error='{s.get('error_message', None)}'")
                logger.debug("--------------------------")
            all_completed = all(s.get("completed", False) for s in states)
            if debug: print(f"--- Finished Generation Step {step_count} (All Completed: {all_completed}) ---")

        if step_count >= max_overall_steps:
            logger.warning(f"Generation stopped after reaching max overall steps ({max_overall_steps}).")
            # Mark any non-completed states as errored/incomplete
            for s in states:
                if not s.get("completed", False):
                    s["completed"] = True
                    s["error_message"] = "Max overall generation steps reached."
                    # Append TASK_ERROR to content if possible and not already terminated
                    if s["messages"] and s["messages"][-1]["role"] == "assistant" and \
                       "<TASK_FINISHED>" not in s["messages"][-1]["content"] and \
                       "<TASK_ERROR>" not in s["messages"][-1]["content"]:
                        s["messages"][-1]["content"] += "\n<TASK_ERROR>Max steps reached.</TASK_ERROR>"
                        # Ideally, re-tokenize and update mask here, but might be complex.
                        # For simplicity, just mark completed. Reward function should handle this.

        logger.info(f"Executing Ground Truth calls for all {len(states)} final states...")
        for i, state in enumerate(states):
            # Ensure environment exists before calling GT
            if "ground_truth_environment" not in state or not state["ground_truth_environment"]:
                 logger.error(f"State {i} (ID: {state.get('id', 'N/A')}): Cannot run GT calls, ground_truth_environment missing.")
                 state["ground_truth_error"] = "GT environment missing at final execution stage."
                 continue # Skip GT for this state

            try:
                # Call the modified, robust call_tool for ground truth
                # Pass None for tool_json_str as it's not used for GT mode
                gt_status_msg, state = self.call_tool(
                    tool_json_str=None,
                    state=state,
                    debug=(debug and i < 2), # Limit debug noise for GT calls
                    ground_truth=True
                )
                if debug and i < 2: # Log status for first few states
                    logger.debug(f"State {i} (ID: {state.get('id', 'N/A')}) GT Execution Status: {gt_status_msg}")
                # Error details are now stored in state["ground_truth_error"] by call_tool

            except Exception as e_gt_final:
                # This catch is a fallback, call_tool should handle internal errors now
                logger.exception(f"State {i} (ID: {state.get('id', 'N/A')}): Unexpected exception during final GT execution call: {e_gt_final}")
                state["ground_truth_error"] = f"Unexpected exception in final GT call: {e_gt_final}"
        logger.info("Finished executing Ground Truth calls.")

        # --- Cleanup and Package Results ---
        self.cleanup_instances()
        output = self._package_generate_output(states)
        return output

    def _package_generate_output(self, states: List[Dict[str, Any]]) -> Dict:
        """Helper to format the final output of the generate method."""
        completion_messages = []
        completion_ids = []
        completion_mask = []

        for i, s in enumerate(states):
            # Extract the full assistant message content (should be one message)
            assistant_msgs_content = [msg["content"] for msg in s.get("messages", [])[s.get("prompt_messages_count", 0):] if msg.get("role") == "assistant"]
            # Store the content as a list containing one dictionary (or empty if no assistant message)
            if assistant_msgs_content:
                 completion_messages.append([{"role": "assistant", "content": assistant_msgs_content[0]}])
            else:
                 completion_messages.append([]) # Handle cases where generation failed early

            c_ids = s.get("completion_ids", [])
            c_mask = s.get("completion_mask", [])

            # Final length check/correction before returning
            if len(c_ids) != len(c_mask):
                logger.error(f"Final output length mismatch for state {i} (ID: {s.get('id', 'N/A')})! IDs: {len(c_ids)}, Mask: {len(c_mask)}. Truncating.")
                min_len = min(len(c_ids), len(c_mask))
                c_ids = c_ids[:min_len]
                c_mask = c_mask[:min_len]

            completion_ids.append(c_ids)
            completion_mask.append(c_mask)


        output = {
            "ids": completion_ids,
            "messages": completion_messages, # List of lists, each inner list usually has one assistant dict
            "mask": completion_mask,
            "states": states, # Return the final states for reward calculation etc.
        }
        return output

    def eval(self, model: Union[str, LLM], batch_size: int = 10, **kwargs: Any):
        # Evaluation logic typically resides in the Trainer using generate and rubrics
        logger.warning("BfclITEnv.eval is basic. Evaluation typically handled by Trainer.")
        if self.eval_dataset is None:
            self.eval_dataset = self.get_eval_dataset(max_num_turns=self.max_num_turns) # Use property

        if self.eval_dataset is None or len(self.eval_dataset) == 0:
             logger.warning("Evaluation dataset is empty or None. Skipping eval.")
             return None, [] # Return None dataset and empty rewards

        logger.info(f"Eval dataset size: {len(self.eval_dataset)}. Returning dataset (evaluation logic external).")
        # Placeholder return, actual evaluation loop is external
        return self.eval_dataset, []
