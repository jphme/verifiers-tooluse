import copy
import gc
import importlib
import inspect
import json
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from datasets import Dataset
from huanzhi_utils import load_file
from loguru import logger
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.envs.tool_env import infer_schema_from_function
from verifiers.parsers import XMLParser
from verifiers.prompts import BFCL_PROMPT
from verifiers.rubrics import BfclRubric
from verifiers.tools.bfcl_tools import INVOLVED_CLASS_TO_FUNC_DOC_PATH
from verifiers.utils import preprocess_dataset

from ..imports import LLM, SamplingParams  # type: ignore


class BfclEnv(MultiStepEnv):
    def __init__(
        self,
        dataset: str = "bfcl",
        tools: List[Callable] = [],
        system_prompt: str = BFCL_PROMPT,
        few_shot: List[Dict[str, str]] = [],
        sampling_args={
            "stop": [
                "</tool>",
                "<TASK_FINISHED>",
                "<TASK_ERROR>",
            ],
            "include_stop_str_in_output": True,
        },
        mask_env_response: bool = True,
        max_num_turns: int = 1,
        max_steps_per_turn: int = 10,
        curriculum_learning: bool = True,
        use_latest_trl: bool = False,
        **kwargs,
    ):
        logger.info("Initializing MultiStepEnv")
        super().__init__(
            system_prompt=system_prompt,
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
        self.env_instances = {}  # Add this line to store instances
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        # print(self.tool_schemas)
        self.tools = {tool.__name__: tool for tool in tools}
        # print(self.tools)

        # Format the system prompt with tool descriptions
        # tool_descriptions = format_tool_descriptions(self.tool_schemas)
        # formatted_prompt = system_prompt.format(tools=tool_descriptions)

        self.dataset_name = dataset
        self.curriculum_learning = curriculum_learning
        # APPARENTLY NOT NEEDED
        # if self.dataset_name == "bfcl":
        #    logger.info(f"Preprocessing dataset {dataset}")
        #    self.dataset = preprocess_dataset(
        #        dataset_name=dataset,
        #        split="train",
        #        system_prompt=system_prompt,
        #        few_shot=few_shot,
        #        curriculum_learning=self.curriculum_learning,
        #    )
        #    self.eval_dataset = None
        # else:
        #    raise Exception("Invalid dataset name")
        self.max_num_turns = max_num_turns
        self.max_steps_per_turn = max_steps_per_turn
        logger.info("Initializing Scoring Rubric")
        self.rubric = BfclRubric()
        logger.info("Initializing LLM + Env Parsers")
        self.llm_parser = XMLParser(fields=["reasoning", "tool"])
        self.env_parser = XMLParser(fields=["tool_result"])
        self.use_latest_trl = use_latest_trl
        self.message_end_id = 151645

    def get_dataset(self, max_num_turns: int = -1, **kwargs: Any) -> Dataset:
        if self.dataset is None:
            self.dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="train",
                system_prompt=self.system_prompt,
                few_shot=self.few_shot,
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
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="test",
                system_prompt=self.system_prompt,
                few_shot=self.few_shot,
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

    def _get_step_count(
        self, messages: List[Dict[str, str]], debug: bool = False
    ) -> int:
        # Find index of last user message
        last_user_idx = -1
        for i, message in enumerate(messages):
            if message.get("role") == "user":
                last_user_idx = i

        # Count assistant messages after last user message
        if last_user_idx != len(messages) - 1:
            step_count = len(
                [
                    message
                    for message in messages[last_user_idx + 1 :]
                    if message.get("role") == "assistant"
                ]
            )
        else:
            step_count = 0
        if debug:
            print(f"Step Count: {step_count}")
        return step_count

    def is_completed(
        self, state: Dict[str, Any] = None, debug: bool = False, **kwargs: Any
    ) -> bool:
        return self.current_entry_completed(state=state, debug=debug)

    def current_entry_completed(
        self, state: Dict[str, Any] = None, debug: bool = False, **kwargs: Any
    ) -> bool:
        messages = state["messages"]
        step_count = self._get_step_count(messages, debug=debug)
        if step_count >= self.max_steps_per_turn:
            if debug:
                print(
                    f"Step count reached max steps per turn which is {self.max_steps_per_turn}"
                )
                time.sleep(3)
            return True

        # Check question bank empty and <TASK_FINISHED>, or <TASK_ERROR> in llm response
        user_question_bank = json.loads(state["dataset_row"]["user_question_bank"])
        llm_response = messages[-1]["content"]
        # If reasoning is present then remove it, only check for TASK_ERROR in the solution part
        if "<reasoning>" in llm_response and "</reasoning>" in llm_response:
            llm_response = llm_response.split("</reasoning>")[1]
        if (
            (
                (len(user_question_bank) == 0)
                and (self.current_turn_completed(state=state, debug=debug))
            )
            or ("TASK_ERROR" in llm_response)
            or ("task_error" in llm_response)
        ):
            if debug:
                if "TASK_ERROR" in llm_response:
                    print("Found TASK_ERROR in response. Current Entry Completed")
                else:
                    print(
                        "No more user questions in question bank. Current Entry Completed"
                    )
                time.sleep(3)
            return True
        else:
            return False

    def current_turn_completed(
        self, state: Dict[str, Any] = None, debug: bool = False, **kwargs: Any
    ) -> bool:
        messages = state["messages"]
        if (
            "TASK_FINISHED" in messages[-1]["content"]
            or "task_finished" in messages[-1]["content"]
        ):
            if debug:
                print("Found TASK_FINISHED in current turn. Current Turn Completed")
                time.sleep(3)
            return True
        return False

    def call_tool(
        self,
        tool_json: str,
        state: Dict[str, Any] = None,
        debug: bool = False,
        ground_truth: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call a tool based on JSON command."""
        if ground_truth:
            if debug:
                print("Executing Ground Truth Tool Call")
                time.sleep(3)
            try:
                if not isinstance(tool_json, list):
                    print(tool_json)
                    raise Exception(
                        "Error in ground truth tool execution is not expected!!"
                    )

                all_func_call_results = []
                # Create mapping of method names to instance names
                method_to_instance = {}
                for class_name, instance in state["ground_truth_environment"].items():
                    for method_name, method in inspect.getmembers(
                        instance, predicate=inspect.ismethod
                    ):
                        if not method_name.startswith("_"):
                            method_to_instance[method_name] = class_name

                # Process each function call
                for func_call in tool_json:
                    # Add the instance reference to the method call
                    if "(" not in func_call:
                        print(tool_json)
                        print(func_call)
                        raise Exception(
                            "Error in ground truth tool execution is not expected!!"
                        )

                    method_name = func_call.split("(")[0].strip()
                    if method_name not in method_to_instance:
                        print(tool_json)
                        print(func_call)
                        raise Exception(
                            "Error in ground truth tool execution is not expected!!"
                        )

                    class_name = method_to_instance[method_name]
                    instance = state["ground_truth_environment"][class_name]
                    modified_call = (
                        f"state['ground_truth_environment']['{class_name}'].{func_call}"
                    )

                    if debug:
                        print(f"Executing ground truth call: {func_call}")
                        time.sleep(3)
                    try:
                        result = eval(modified_call)
                        result_str = str(result) if result is not None else "Success"
                        all_func_call_results.append(
                            f"Function Call {func_call} Succeeded. Result: {result_str}"
                        )
                    except Exception:
                        print(tool_json)
                        print(func_call)
                        raise Exception(
                            "Error in ground truth tool execution is not expected!!"
                        )
                return json.dumps(all_func_call_results), state
            except Exception as e:
                print(tool_json)
                print(e)
                raise Exception(
                    "Error in ground truth tool execution is not expected!!"
                )

        # Handling model tool calls
        try:
            command = json.loads(tool_json)
            all_func_call_results = []
            # Process tool calls one by one, if later tool call fails, previous successful tool calls are still executed
            if not isinstance(command, list):
                all_func_call_results.append(
                    "Error: Invalid tool command. Tool command must be one list of JSON objects. Please ensure correct formatting."
                )
                return json.dumps(all_func_call_results), state
            if command == []:
                all_func_call_results.append(
                    "Function Call Failed. Error: Found empty tool calls."
                )
                return json.dumps(all_func_call_results), state
            for tool_call in command:
                # Check if tool_call is a dictionary with 'name' and 'args' keys and 'args' is a dictionary
                if not (
                    isinstance(tool_call, dict)
                    and "name" in tool_call
                    and "args" in tool_call
                    and isinstance(tool_call["args"], dict)
                ):
                    all_func_call_results.append(
                        f"Function Call {tool_call} Failed. Error: Tool command must be a dictionary with 'name' key and 'args' as a dictionary. Function calls after this will not be executed."
                    )
                    return json.dumps(all_func_call_results), state

                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                for key, value in tool_args.items():
                    if isinstance(value, list):
                        tool_args[key] = tuple(value)
                tool_call["args"] = tool_args
                if debug:
                    print(f"Tool Name: {tool_name}")
                    print(f"Tool Args: {tool_args}")
                    time.sleep(3)

                # Check if tool_name exists as a method in any class instance
                found_method = False
                if self.env_instances == {}:
                    raise Exception("Environment is empty")
                for class_instance in state["environment"].values():
                    if hasattr(class_instance, tool_name):
                        found_method = True
                        if debug:
                            print(
                                f"Found method {tool_name} in class {class_instance.__class__.__name__}"
                            )
                        tool_func = getattr(class_instance, tool_name)
                        break

                if not found_method:
                    available_tools = []
                    for class_name in state["dataset_row"]["involved_classes"]:
                        func_doc = load_file(
                            INVOLVED_CLASS_TO_FUNC_DOC_PATH[class_name]
                        )
                        for func in func_doc:
                            available_tools.append(func["name"])
                    if tool_name in available_tools:
                        print(f"Tool Name: {tool_name}")
                        print(
                            f"Involved Classes: {state['dataset_row']['involved_classes']}"
                        )
                        print(f"State Environment: {state['environment']}")
                        raise Exception(
                            f"Error: Method '{tool_name}' found in involved classes but not found in any class instance. Available Tools: {available_tools}"
                        )
                    all_func_call_results.append(
                        f"Function Call {tool_call} Failed. Error: Method '{tool_name}' not found in any class instance. Function calls after this will not be executed."
                    )
                    return json.dumps(all_func_call_results), state

                # Call the tool function with arguments using the class instance
                try:
                    result = tool_func(**tool_args)
                except Exception as e:
                    all_func_call_results.append(
                        f"Function Call {tool_call} Failed during execution. Error: {e}. Function calls after this will not be executed."
                    )
                    return json.dumps(all_func_call_results), state

                # If function call succeeds but tool result is error
                # NOTE: This below gives false negatives: sometimes the function return result has the word "error" in it
                # if "error" in str(result).lower():
                if "'error':" in str(result).lower():
                    all_func_call_results.append(
                        f"Function Call {tool_call} Failed during execution. Error: {result}. Function calls after this will not be executed."
                    )
                    return json.dumps(all_func_call_results), state

                # Otherwise, the function call is successful
                all_func_call_results.append(
                    f"Function Call {tool_call} Succeeded. Result: {result}"
                )
                state["successful_func_calls"][-1].append(tool_call)

            return json.dumps(all_func_call_results), state
        except json.JSONDecodeError:
            all_func_call_results = []
            all_func_call_results.append(
                "Error in decoding tool call: Invalid JSON format. Tool command must be one list of JSON objects. Please ensure correct formatting."
            )
            return json.dumps(all_func_call_results), state
        except Exception as e:
            print(tool_json)
            print(e)
            raise Exception(f"Error here is not expected!! Error: {e}")

    def env_response(
        self, state: Dict[str, Any] = None, debug: bool = False, **kwargs: Any
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        messages = state["messages"]
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            if debug:
                print(f"Parsed: {parsed}")
                time.sleep(3)
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, "tool") and parsed.tool is not None:
                result, state = self.call_tool(parsed.tool, state=state, debug=debug)
                if len(result) > 0:
                    tool_result = f"<tool_result> {result} </tool_result>"
                    return {"role": "system", "content": tool_result}, state
                else:
                    all_func_call_results = [
                        "Error: Tool execution returned empty output."
                    ]
                    tool_result = f"<tool_result> {json.dumps(all_func_call_results)} </tool_result>"
                    return {"role": "system", "content": tool_result}, state
            else:
                all_func_call_results = [
                    "Error: Function call not found in current assistant response. Tool command must be one list of JSON objects. Please ensure correct formatting. If task is finished, please respond with the <TASK_FINISHED> tag. If task is problematic, please respond with the <TASK_ERROR> tag."
                ]
                tool_result = (
                    f"<tool_result> {json.dumps(all_func_call_results)} </tool_result>"
                )
                return {"role": "system", "content": tool_result}, state
        except Exception as e:
            if "not expected" in str(e).lower():
                raise Exception(f"Error in env_response is not expected!! Error: {e}")
            all_func_call_results = [
                "Error: Invalid XML format: {str(e)}.  Tool command must be one list of JSON objects. Please ensure correct formatting."
            ]
            tool_result = (
                f"<tool_result> {json.dumps(all_func_call_results)} </tool_result>"
            )
            return {"role": "system", "content": tool_result}, state

    def step(
        self,
        states: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]
        # NOTE: llm_responses is a list of ChatCompletion (rollouts)
        if debug:
            if not states[0]["completed"]:
                print(
                    "--------------------------------New round of generation--------------------------------"
                )
                print(
                    f"Latest Input Message to LLM: {messages_to_step[0][-2:] if len(messages_to_step[0]) >= 2 else messages_to_step[0]}"
                )
                time.sleep(3)
        # NOTE: If TRL version is 0.16.0, use vllm_client.generate instead (Under Construction)
        if self.use_latest_trl:
            raise Exception("Doesn't Support TRL 0.16.0 yet")
            # llm_responses = self.vllm_client.generate(
            #     prompts=messages_to_step,
            #     # n=self.num_generations,
            #     repetition_penalty=self.repetition_penalty,
            #     temperature=self.temperature,
            #     top_p=self.top_p,
            #     top_k=-1 if self.top_k is None else self.top_k,
            #     min_p=0.0 if self.min_p is None else self.min_p,
            #     max_tokens=self.max_completion_length,
            #     guided_decoding_regex=self.guided_decoding_regex,
            # )
        else:
            llm_responses = llm.chat(
                messages_to_step, sampling_params=sampling_params, use_tqdm=False
            )  # type: ignore

        for i, j in enumerate(live_indices):
            if len(states[j]["prompt_ids"]) == 0:
                states[j]["prompt_ids"] = llm_responses[i].prompt_token_ids
            llm_response = llm_responses[i].outputs[0].text
            if "<reasoning>" in llm_response and "</reasoning>" in llm_response:
                llm_response_without_reasoning = llm_response.split("</reasoning>")[1]
                states[j]["multi_turn_history"].append(
                    {"role": "assistant", "content": llm_response_without_reasoning}
                )
            else:
                states[j]["multi_turn_history"].append(
                    {"role": "assistant", "content": llm_response}
                )
            states[j]["messages"].append({"role": "assistant", "content": llm_response})
            # get token lengths of env response and new completion
            total_prev_len = len(states[j]["prompt_ids"]) + len(
                states[j]["completion_ids"]
            )
            env_response_len = (
                len(list(llm_responses[i].prompt_token_ids)) - total_prev_len
            )  # type: ignore
            new_completion_len = len(llm_responses[i].outputs[0].token_ids)
            if debug:
                if j == 0:
                    print(f"llm_response: {states[j]['messages'][-1]['content']}")
                    time.sleep(3)

            # update completion masks
            states[j]["completion_mask"].extend([self.env_mask] * env_response_len)
            states[j]["completion_mask"].extend([1] * new_completion_len)

            # update completion ids
            states[j]["completion_ids"] = list(llm_responses[i].prompt_token_ids)  # type: ignore
            states[j]["completion_ids"].extend(
                list(llm_responses[i].outputs[0].token_ids)
            )
            states[j]["completion_ids"] = states[j]["completion_ids"][
                len(states[j]["prompt_ids"]) :
            ]

            if "successful_func_calls" not in states[j]:
                states[j]["successful_func_calls"] = [[]]

            if len(states[j]["completion_ids"]) > len(states[j]["completion_mask"]):  # type: ignore
                states[j]["completion_mask"].extend(
                    [1]
                    * (
                        len(states[j]["completion_ids"])
                        - len(states[j]["completion_mask"])
                    )
                )  # type: ignore
            if len(states[j]["completion_mask"]) > len(states[j]["completion_ids"]):  # type: ignore
                states[j]["completion_mask"] = states[j]["completion_mask"][
                    : len(states[j]["completion_ids"])
                ]  # type: ignore

            if (
                self.current_entry_completed(state=states[j], debug=(debug and j == 0))
                or len(states[j]["completion_ids"]) > sampling_params.max_tokens - 1
            ):  # type: ignore
                states[j]["completed"] = True
                states[j]["completion_ids"] = states[j]["completion_ids"][
                    : sampling_params.max_tokens
                ]
                states[j]["completion_mask"] = states[j]["completion_mask"][
                    : len(states[j]["completion_ids"])
                ]

                # Clearing up the ground truth answer bank
                while (
                    len(json.loads(states[j]["dataset_row"]["ground_truth_bank"])) > 0
                ):
                    if debug:
                        if j == 0:
                            print("Clearing up the ground truth answer bank")
                            time.sleep(3)
                    ground_truth_answer_bank = json.loads(
                        states[j]["dataset_row"]["ground_truth_bank"]
                    )
                    ground_truth_answer = ground_truth_answer_bank.pop(0)
                    states[j]["dataset_row"]["ground_truth_bank"] = json.dumps(
                        ground_truth_answer_bank
                    )
                    _, states[j] = self.call_tool(
                        ground_truth_answer,
                        state=states[j],
                        debug=(debug and (j == 0)),
                        ground_truth=True,
                    )
                    if (
                        len(json.loads(states[j]["dataset_row"]["ground_truth_bank"]))
                        > 0
                    ):
                        states[j]["successful_func_calls"].append([])
                assert len(states[j]["successful_func_calls"]) == len(
                    json.loads(states[j]["dataset_row"]["answer"])
                )
            elif self.current_turn_completed(
                state=states[j], debug=(debug and (j == 0))
            ):
                # This is when <TASK_FINISHED> is found, give the next user question from user question bank
                # NOTE: user_question_bank is a list of lists
                user_question_bank = json.loads(
                    states[j]["dataset_row"]["user_question_bank"]
                )
                next_user_question = user_question_bank.pop(0)[0]["content"]
                states[j]["dataset_row"]["user_question_bank"] = json.dumps(
                    user_question_bank
                )
                states[j]["messages"].append(
                    {"role": "user", "content": next_user_question}
                )
                states[j]["multi_turn_history"].append(
                    {"role": "user", "content": next_user_question}
                )

                ground_truth_answer_bank = json.loads(
                    states[j]["dataset_row"]["ground_truth_bank"]
                )
                if len(ground_truth_answer_bank) > 0:
                    ground_truth_answer = ground_truth_answer_bank.pop(0)
                    states[j]["dataset_row"]["ground_truth_bank"] = json.dumps(
                        ground_truth_answer_bank
                    )
                _, states[j] = self.call_tool(
                    ground_truth_answer,
                    state=states[j],
                    debug=(debug and (j == 0)),
                    ground_truth=True,
                )

                # Append a new list in successful_func_calls for the next turn
                states[j]["successful_func_calls"].append([])
                if debug:
                    if j == 0:
                        print(f"next_user_question: {next_user_question}")
                        time.sleep(3)
            else:
                env_response, states[j] = self.env_response(
                    state=states[j], debug=(debug and (j == 0))
                )
                states[j]["messages"].append(env_response)
                states[j]["multi_turn_history"].append(env_response)
                if debug:
                    if j == 0:
                        print(f"env_response: {states[j]['messages'][-1]['content']}")
                        time.sleep(3)
            # enforce that the completion mask and completion ids are the same length
            # weird bug that happens rarely and only for certain models; something tokenizer related :(
            if not len(states[j]["completion_mask"]) == len(
                states[j]["completion_ids"]
            ):
                print(states[j]["messages"])
                print(states[j]["completion_mask"])
                print(states[j]["completion_ids"])
                min_len = min(
                    len(states[j]["completion_mask"]), len(states[j]["completion_ids"])
                )
                states[j]["completion_mask"] = states[j]["completion_mask"][:min_len]
                states[j]["completion_ids"] = states[j]["completion_ids"][:min_len]

            if len(states[j]["completion_mask"]) != len(states[j]["completion_ids"]):
                print(f"Completion mask: {states[j]['completion_mask']}")
                print(f"Completion ids: {states[j]['completion_ids']}")
                print(f"State: {states[j]}")
            assert len(states[j]["completion_mask"]) == len(
                states[j]["completion_ids"]
            ), (
                f"Completion mask and completion ids length mismatch: {len(states[j]['completion_mask'])} != {len(states[j]['completion_ids'])}"
            )
        return states

    def _initialize_environments(self, states: List[Dict[str, Any]]) -> None:
        """Initialize environment instances for the given states"""
        for i in range(len(states)):
            if "instance_id" not in states[i]:
                states[i]["instance_id"] = id(states[i])
            instance_id = states[i]["instance_id"]

            if instance_id not in self.env_instances:
                self.env_instances[instance_id] = {}

            involved_classes = states[i]["dataset_row"]["involved_classes"]
            if "environment" not in states[i]:
                states[i]["environment"] = {}
                states[i]["ground_truth_environment"] = {}
                states[i]["initial_environment"] = {}

            for class_name in involved_classes:
                if class_name not in states[i]["environment"]:
                    # Import and initialize the class
                    module_name = self.CLASS_FILE_PATH_MAPPING[class_name]
                    module = importlib.import_module(module_name)
                    class_ = getattr(module, class_name)
                    class_instance = class_()
                    ground_truth_class_instance = class_()
                    initial_instance_copy = class_()

                    # Configure non-stateless classes
                    if class_name not in self.STATELESS_CLASSES:
                        initial_config = json.loads(
                            states[i]["dataset_row"]["initial_config"]
                        )
                        class_initial_config = initial_config.get(class_name, {})
                        class_instance._load_scenario(
                            copy.deepcopy(class_initial_config)
                        )
                        ground_truth_class_instance._load_scenario(
                            copy.deepcopy(class_initial_config)
                        )
                        initial_instance_copy._load_scenario(
                            copy.deepcopy(class_initial_config)
                        )

                    # Store instances in instance dictionary
                    self.env_instances[instance_id][class_name] = {
                        "main": class_instance,
                        "ground_truth": ground_truth_class_instance,
                        "initial_instance": initial_instance_copy,
                    }
                    states[i]["environment"][class_name] = class_instance
                    states[i]["ground_truth_environment"][class_name] = (
                        ground_truth_class_instance
                    )
                    states[i]["initial_environment"][class_name] = initial_instance_copy
        return states

    def cleanup_instances(self) -> None:
        """Clean up all environment instances"""
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
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)
        # initialize state variables
        all_completed = False
        states = [
            {
                "messages": copy.deepcopy(m),
                "multi_turn_history": copy.deepcopy(m),
                "prompt_messages": len(m),
                "prompt_ids": [],
                "completed": False,
                "completion_ids": [],
                "completion_mask": [],
                "dataset_row": copy.deepcopy(dataset_rows[i]),
            }
            for i, m in enumerate(prompts)
        ]
        assert len(states[0]["messages"]) == 2
        if debug:
            print(f"System Prompt: {states[0]['messages'][0]['content']}")
            print(f"User Prompt: {states[0]['messages'][1]['content']}")
            print(f"Number of Rollouts: {len(states)}")
            time.sleep(3)
        # main loop
        states = self._initialize_environments(states)
        while not all_completed:
            states = self.step(states, llm, custom_sp, debug=debug)
            all_completed = all(state["completed"] for state in states)

        completion_messages = [s["messages"][s["prompt_messages"] :] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
            "states": states,
        }

        return output

    def eval(self, model: Union[str, LLM], batch_size: int = 10, **kwargs: Any):
        if self.eval_dataset is None:
            self.eval_dataset = self.get_eval_dataset(max_num_turns=self.max_num_turns)

        rewards = []
        return self.eval_dataset, rewards
