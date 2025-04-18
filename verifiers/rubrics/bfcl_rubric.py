from typing import List, Dict, Any
import ast
import json
import time
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from typing import Dict
from pydantic import BaseModel, Field
# from bespokelabs import curator
from datasets import Dataset
import os
import openai

# os.environ["CURATOR_DISABLE_CACHE"] = "1"
# os.environ["CURATOR_VIEWER"] = "0"

# BFCL_PROMPT = """\
# You are an expert in composing functions. You are given a question from a user and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to complete the task.
# You have access to the following tools to help solve the task:

# {tools}

# For each step:
# 1. Start with a step-by-step thinking process inside <reasoning> </reasoning> tags to think through the problem.
# 2. If needed, use tools by writing one or more JSON commands as a list inside <tool> </tool> tags. Each item in the list should have a name and args key, with args being a dictionary.
#    example: <tool> [{{"name": func_1_name, "args": {{arg1: value1, arg2: value2}}}}, {{"name": func_2_name, "args": {{arg3: value3, arg4: value4}}}}] </tool>
#    Tools expect specific JSON input formats. Do not make up tools or arguments that aren't listed.
# 3. After you have used the tools, you will see the tool outputs inside <tool_result> </tool_result> tags in the same order from the system.
# 4. If you believe the current task is completed and no more tool, summarize your progresses and output <TASK_FINISHED> in the end of your response to terminate the conversation.
# 5. Otherwise if you believe the task is not able to be completed, summarize what is problematic and output <TASK_ERROR> in the end of your response to terminate the conversation.
# """

# class JudgeResult(BaseModel):
#     is_gibberish: bool = Field(description="Whether the response contains gibberish.")
#     # reasoning: str = Field(description="Reasoning for the judgment.")

# class Judge(curator.LLM):
#     response_format = JudgeResult

#     def prompt(self, input: Dict) -> str:
#         model_completion = [completion for completion in input['completion'] if completion['role'] == 'assistant']
#         # model_completion = input['completion']
# #         prompt = f'''Determine whether the following tool-calling agent trajectory contains gibberish output or useless repetitions.

# # The tool-calling agent follows these rules:
# # {BFCL_PROMPT}

# # When evaluating for gibberish, consider:
# # 1. Gibberish includes: random tokens, completely irrelevant text, nonsensical outputs, or text that doesn't follow the expected format.
# # 2. NOT gibberish: Proper and accurate use of <reasoning>, <tool>, and <TASK_FINISHED>/<TASK_ERROR> tags according to the rules.
# # 3. Make sure to distinguish between gibberish output from the model itself or the tool results. For example, if the tool results looks gibberish, and the model is just reporting it, it is not the model's fault.

# # Analyze the following trajectory, pay attention to assistant messages only and ignore system messages:
# # {model_completion}
# # '''
#         prompt = f"Determine whether the following tool-calling agent responses contains gibberish output or useless repetitions: {model_completion}"
#         return prompt
        
#     def parse(self, input: Dict, response: JudgeResult) -> Dict:
#         input['is_gibberish_judge'] = response.is_gibberish
#         # input['judge_reasoning'] = response.reasoning
#         return input

# judge = Judge(model_name="gpt-4o-mini", 
#               generation_params={"temperature": 0.0})

class BfclRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", "tool"]),
                 env_parser: XMLParser = XMLParser(fields=["tool_result"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            # self.tool_execution_reward_func,
            self.unified_reward_func,
        ]
        # self.llm_judge = Judge(model_name="gpt-4o-mini", 
        #                      generation_params={"temperature": 0.1})

    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], states: List[Dict[str, Any]], 
                                   debug: bool = False, max_score: float = 0.2) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        if debug:
            print(f"Computing Tool Execution Reward\n")
            time.sleep(3)
        def check_execution(trajectory, debug=debug):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    if debug:
                        print(f"LLM Response: {msg['content']}\n")
                        time.sleep(3)
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'system':
                            if debug:
                                print(f"Found properly formatted tool message: {parsed.tool}\n")
                                time.sleep(3)
                            # Check response with env_parser
                            parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                            if hasattr(parsed_response, 'tool_result') and parsed_response.tool_result is not None:
                                try:
                                    tool_results = json.loads(parsed_response.tool_result)
                                except json.JSONDecodeError as e: # NOTE: This means parser malfunctions due to potentially double tags
                                    tool_results = []
                                except Exception as e:
                                    print(f"Tool Result: {parsed_response.tool_result}")
                                    raise Exception(f"Error in Parsing Tool Result is Not Expected!! Error: {e}")
                                for tool_result in tool_results:
                                    tool_attempts += 1
                                    if not "error" in tool_result.lower():
                                        successful_executions += 1
                                        if debug:
                                            print(f"Successful execution: {tool_result}\n")
                                            time.sleep(3)
                                    else:
                                        if debug:
                                            print(f"Error in execution: {tool_result}\n")
                                            time.sleep(3)

                            # if hasattr(parsed_response, 'tool_result') and parsed_response.tool_result is not None and not "error" in parsed_response.tool_result.lower():
                            #     successful_executions += 1
                            #     if debug:
                            #         print(f"Successful execution: {parsed_response.tool_result}")
                            #         time.sleep(3)
                            # else:
                            #     successful_executions += 0
                            #     if debug:
                            #         print(f"Error in execution: {parsed_response.tool_result}")
                            #         time.sleep(3)
            if debug:
                print(f"Successful executions: {successful_executions}")
                print(f"Tool attempts: {tool_attempts}\n")
                time.sleep(3)
            # Calculate reward
            if tool_attempts == 0:
                if debug:
                    print(f"Found no tool calls in the trajectory\n")
                    time.sleep(3)
                return 0.0
            final_score = max_score * (successful_executions / tool_attempts) if tool_attempts > 0 else 0.0
            if debug:
                print(f"Final Tool Execution Score: {final_score}\n")
                time.sleep(3)
            return final_score
        
        return [check_execution(c, debug=(debug and (j==0))) for j, c in enumerate(completions)]

    @staticmethod
    def _parse_function_call(func_call_str: str):
        """
        Parses a function call string into a JSON-like dictionary.
        
        :param func_call_str: String representation of a function call.
        :return: JSON-like dictionary with function name and arguments.
        """
        try:
            # Parse the function call string into an AST node
            tree = ast.parse(func_call_str, mode='eval')

            # Ensure it is a function call
            if not isinstance(tree.body, ast.Call):
                raise ValueError("Input is not a valid function call.")

            # Extract function name
            func_name = tree.body.func.id if isinstance(tree.body.func, ast.Name) else None
            if not func_name:
                raise ValueError("Could not determine function name.")

            # Extract arguments
            args_dict = {}

            # Handle keyword arguments (named parameters)
            for kw in tree.body.keywords:
                args_dict[kw.arg] = ast.literal_eval(kw.value)  # Convert AST to actual Python value

            # Handle positional arguments (if any, though your example has none)
            for i, arg in enumerate(tree.body.args):
                args_dict[f"arg{i+1}"] = ast.literal_eval(arg)

            # Create JSON output
            json_obj = {
                "name": func_name,
                "args": args_dict
            }

            return json_obj

        except Exception:
            raise Exception(f"Error in Parsing Ground Truth Function Call is Not Expected!!")

    @staticmethod
    def _is_subsequence_unordered(list1, list2) -> tuple[bool, list]:
        """
        Checks if all elements of list1 are present in list2, regardless of order.
        Also returns the elements of list1 that are not present in list2.
        """
        if list1 == [] or list2 == []:
            return False, []
        # Copy list2 to avoid modifying the original list during checks
        list2_copy = list2[:]
        
        # Check each item in list1 to see if it exists in list2_copy
        missing_elements = []
        for item in list1:
            try:
                # Attempt to remove one occurrence of `item` from list2_copy to handle duplicates
                list2_copy.remove(item)
            except ValueError:
                # If item is not found, add it to missing_elements
                missing_elements.append(item)
        
        # If there are missing elements, list1 is not a subsequence of list2
        is_subsequence = len(missing_elements) == 0
        return is_subsequence, missing_elements

    @staticmethod
    def compare_instances(model_obect, ground_truth_object):
        """
        Checks if the model_object has the same attributes as the ground_truth_object. They are instances of the same class.
        """
        assert type(model_obect) == type(
            ground_truth_object
        ), "Objects are not of the same type."
        differences = {}
        valid = True
        for attr_name in vars(ground_truth_object):
            # We don't check for private attributes
            if attr_name.startswith("_"):
                continue
            model_attr = getattr(model_obect, attr_name)
            ground_truth_attr = getattr(ground_truth_object, attr_name)

            if model_attr != ground_truth_attr:
                valid = False
                differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}

        return valid, differences
    
    def _check_gibberish(self,completions: List[List[Dict[str, str]]], debug: bool = False) -> bool:
        """
        Checks if the completions contain gibberish output.
        """

        model_responses = Dataset.from_list([{'completion': [one_response for one_response in c if one_response['role'] == 'assistant']} for c in completions])
        gibberish_results = self.llm_judge(model_responses)
        gibberish_reward = [result['is_gibberish_judge'] for result in gibberish_results]
        return gibberish_reward

    @staticmethod
    def _check_tool_result_occurrence(completions: List[List[Dict[str, str]]], debug: bool = False) -> bool:
        """
        Checks if the tool result occurs in the completions.
        """
        tool_result_occurrences = []
        for completion in completions:
            if any([(msg['role'] == 'assistant' and ('<tool_result>' in msg['content'].lower() or '</tool_result>' in msg['content'].lower())) for msg in completion]):
                tool_result_occurrences.append(True)
            else:
                tool_result_occurrences.append(False)
        return tool_result_occurrences


    def unified_reward_func(self, completions: List[List[Dict[str, str]]], states: List[Dict[str, Any]], 
                          debug: bool = False, 
                          func_match_max_score: float = 0.5, state_match_max_score: float = 0.5, 
                          format_max_score: float = 0.1) -> List[float]:
        """
        Combined reward function that checks state matches, function call matches, and format.
        State and function matches contribute 0.5 each to base score.
        If base score is perfect, format check can add 0.1 more.
        """
        if debug:
            print(f"Computing Unified Reward\n")
            time.sleep(3)

        def check_unified(trajectory, state, debug=debug):
            # First check state matches
            if debug:
                print(f"Checking State Matches\n")
                time.sleep(3)
            num_state_matches = 0
            num_state_total = 0
            for key in state["ground_truth_environment"]:
                if debug:
                    print(f"Comparing {key} in ground truth and environment")
                    print("Current Environment Attributes:")
                    for attr_name, value in vars(state['environment'][key]).items():
                        if not attr_name.startswith('_'):
                            print(f"  {attr_name}: {value}")
                    print("\nGround Truth Environment Attributes:")
                    for attr_name, value in vars(state['ground_truth_environment'][key]).items():
                        if not attr_name.startswith('_'):
                            print(f"  {attr_name}: {value}")
                    time.sleep(3)

                valid, diffs = self.compare_instances(state["ground_truth_environment"][key], state["environment"][key])
                if debug:
                    print(f"State Match: {valid}")
                    print(f"Differences: {diffs}")
                    time.sleep(3)
                num_state_matches += int(valid)
                num_state_total += 1
                
            state_score = state_match_max_score * (num_state_matches / num_state_total)
            if debug:
                print(f"State Score: {state_score}\n")
                time.sleep(3)

            # Then check function calls
            if debug:
                print(f"Checking Function Calls\n")
                time.sleep(3)
            num_func_matches = 0
            num_func_total = 0
            model_func_calls = state["successful_func_calls"]
            ground_truth_func_calls = json.loads(state['dataset_row']['answer'])
            assert len(model_func_calls) == len(ground_truth_func_calls)

            for model_calls, gt_calls_str in zip(model_func_calls, ground_truth_func_calls):
                gt_calls = [self._parse_function_call(call_str) for call_str in gt_calls_str]
                
                def make_hashable(value):
                    if isinstance(value, dict):
                        return frozenset((k, make_hashable(v)) for k, v in value.items())
                    elif isinstance(value, list):
                        return tuple(make_hashable(item) for item in value)
                    elif isinstance(value, set):
                        return frozenset(make_hashable(item) for item in value)
                    return value

                comparable_model_calls = [
                    (call["name"], frozenset((k, make_hashable(v)) for k, v in call["args"].items()))
                    for call in model_calls
                ]
                
                for call in gt_calls:
                    if "args" in call:
                        for key, value in call["args"].items():
                            if isinstance(value, list):
                                call["args"][key] = tuple(value)
                    else:
                        raise Exception("Error in Parsing Ground Truth Function Call is Not Expected!!")

                comparable_gt_calls = [
                    (call["name"], frozenset((k, make_hashable(v)) for k, v in call["args"].items()))
                    for call in gt_calls
                ]
                if debug:
                    print(f"Comparable Model Calls: {comparable_model_calls}")
                    print(f"Comparable Ground Truth Calls: {comparable_gt_calls}")
                    time.sleep(3)

                is_match, _ = self._is_subsequence_unordered(comparable_gt_calls, comparable_model_calls)
                if debug:
                    print(f"Is Subsequence: {is_match}")
                    time.sleep(3)
                num_func_matches += int(is_match)
                num_func_total += 1
            func_score = func_match_max_score * (num_func_matches / num_func_total)
            if debug:
                print(f"Function Call Score: {func_score}\n")
                time.sleep(3)
            
            #if base_score != state_match_max_score + func_match_max_score:
            #    if debug:
            #        print(f"Base Score is not perfect, so giving 0 score, and no format #check.\n")
            #        time.sleep(3)
            #    format_score = 0
            #    base_score = 0
            # Only check format if base score is perfect
            #else:
            valid_messages = 0
            total_messages = 0
            for msg in trajectory:
                if msg['role'] == 'assistant':
                    if debug:
                        print(f"Checking Message: {msg['content']}")
                        time.sleep(3)
                    total_messages += 1
                    parsed = self.parser.parse(msg['content'])
                    if debug:
                        print(f"Parsed: {parsed}")
                        time.sleep(3)
                    # Must have either tool content or task status or reasoning for all messages
                    if (hasattr(parsed, 'tool') and parsed.tool is not None) or (hasattr(parsed, 'reasoning') and parsed.reasoning is not None) or ("<TASK_FINISHED>" in msg['content'] or "<TASK_ERROR>" in msg['content']):
                        valid_messages += 1
                        if debug:
                            print(f"Valid: True")
                            time.sleep(3)
                    else:
                        if debug:
                            print(f"Valid: False")
                            time.sleep(3)

            if valid_messages == total_messages:
                format_score = format_max_score
            else:
                format_score = format_max_score * (valid_messages / total_messages) if total_messages > 0 else 0
            base_score = state_score + func_score + format_score

            #NOTE: Experimenting with adding format score to base score
            # base_score += format_score

            # if debug:
            #     print(f"State Score: {state_score}")
            #     print(f"Function Call Score: {func_score}")
            #     print(f"Format Score: {format_score}")
            #     print(f"Final Unified Score: {base_score}")
            #     time.sleep(3)
            return base_score

        unified_rewards = [check_unified(c, s, debug=(debug and (j==0))) for j, (c, s) in enumerate(zip(completions, states))]

        # LLM Judge to determine if gibberish output
        # gibberish_rewards = self._check_gibberish(completions)

        # assert len(gibberish_rewards) == len(unified_rewards)
        # # If gibberish, give 0 reward
        # for i in range(len(gibberish_rewards)):
        #     if gibberish_rewards[i]:
        #         unified_rewards[i] = -1

        # tool_result_occurrences = self._check_tool_result_occurrence(completions)
        # assert len(tool_result_occurrences) == len(unified_rewards)
        # for i in range(len(tool_result_occurrences)):
        #     if tool_result_occurrences[i]:
        #         unified_rewards[i] = -1

        return unified_rewards
