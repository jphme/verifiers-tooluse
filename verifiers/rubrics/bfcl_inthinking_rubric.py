# bfcl_inthinking_rubric.py

import ast
import json
import re
import time
from typing import Any, Dict, List

from loguru import logger

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric


class BfclITRubric(Rubric):
    def __init__(
        self,
        parser: XMLParser = XMLParser(fields=["think", "tool"]),
        env_parser: XMLParser = XMLParser(fields=["tool_result"]),
    ):
        self.parser = parser
        self.env_parser = env_parser
        # Store individual functions internally
        self._reward_components = {
            "unified": self.unified_success_reward_func,
            "tool_exec": self.tool_execution_reward_func,
            "format": self.format_reward_func,
            "self_correct": self.self_correction_reward_func,
        }

    # --- Keep the individual reward functions exactly as defined in the previous step ---
    @staticmethod
    def _parse_function_call(func_call_str: str) -> Dict | None:
        # ... (implementation from previous step) ...
        try:
            tree = ast.parse(func_call_str.strip(), mode='eval')
            if not isinstance(tree.body, ast.Call):
                # logger.warning(f"Could not parse '{func_call_str}' as function call.")
                return None

            func_node = tree.body.func
            if isinstance(func_node, ast.Name):
                func_name = func_node.id
            elif isinstance(func_node, ast.Attribute):
                func_name = func_node.attr
            else:
                # logger.warning(f"Unsupported function call structure in '{func_call_str}'.")
                return None

            args_dict = {}
            for kw in tree.body.keywords:
                try:
                    # Ensure the value is evaluated safely
                    value = ast.literal_eval(kw.value)
                    args_dict[kw.arg] = value
                except ValueError:
                    # logger.warning(f"Could not literal_eval keyword arg value in '{func_call_str}'. Skipping arg '{kw.arg}'.")
                    # Fallback for non-literal values if needed, e.g., represent as string
                    # args_dict[kw.arg] = ast.dump(kw.value) # Or skip
                    pass
                except Exception as e:
                    logger.warning(f"Error evaluating keyword arg '{kw.arg}' in '{func_call_str}': {e}")

            for i, arg in enumerate(tree.body.args):
                try:
                    args_dict[f"pos_arg_{i}"] = ast.literal_eval(arg)
                except ValueError:
                    # logger.warning(f"Could not literal_eval positional arg {i} in '{func_call_str}'. Skipping.")
                    pass
                except Exception as e:
                    logger.warning(f"Error evaluating positional arg {i} in '{func_call_str}': {e}")

            # Normalize args: Convert lists to tuples
            for key, value in args_dict.items():
                if isinstance(value, list):
                    args_dict[key] = tuple(value)

            return {"name": func_name, "args": args_dict}

        except Exception as e:
            # logger.error(f"Failed to parse ground truth function call '{func_call_str}': {e}")
            return None


    @staticmethod
    def compare_instances(model_object, ground_truth_object):
        # ... (implementation from previous step) ...
        if type(model_object) != type(ground_truth_object):
            # logger.warning(f"Type mismatch in compare_instances: {type(model_object)} vs {type(ground_truth_object)}")
            return False

        match = True
        for attr_name in vars(ground_truth_object):
            if attr_name.startswith("_"):
                continue
            if not hasattr(model_object, attr_name):
                # logger.debug(f"Attribute '{attr_name}' missing in model object.")
                match = False
                break
            model_attr = getattr(model_object, attr_name)
            ground_truth_attr = getattr(ground_truth_object, attr_name)
            if model_attr != ground_truth_attr:
                # logger.debug(f"Attribute mismatch for '{attr_name}': Model='{model_attr}' ({type(model_attr)}), GT='{ground_truth_attr}' ({type(ground_truth_attr)})")
                match = False
                break
        return match

    def unified_success_reward_func(
        self,
        completions: List[List[Dict[str, str]]],
        states: List[Dict[str, Any]],
        debug: bool = False,
    ) -> List[float]:
        # ... (implementation from previous step) ...
        rewards = []
        for i, state in enumerate(states):
            completion = completions[i]
            if not completion or completion[0].get("role") != "assistant":
                #  logger.warning(f"State {i}: No valid assistant completion found for unified reward.")
                 rewards.append(0.0)
                 continue

            if debug: logger.debug(f"\n--- Unified Reward Check (State {i}) ---")

            all_states_match = True
            if "ground_truth_environment" not in state or "environment" not in state:
                # logger.warning(f"State {i}: Missing environment or ground_truth_environment for state comparison.")
                all_states_match = False
            else:
                if state["ground_truth_environment"].keys() != state["environment"].keys():
                    #  logger.warning(f"State {i}: Environment keys mismatch. GT: {state['ground_truth_environment'].keys()}, Model: {state['environment'].keys()}")
                     all_states_match = False
                else:
                    for key in state["ground_truth_environment"]:
                        if key not in state["environment"]:
                            #  logger.warning(f"State {i}: Key '{key}' missing in model environment.")
                             all_states_match = False
                             break
                        if debug: logger.debug(f"Comparing state for class: {key}")
                        match = self.compare_instances(
                            state["environment"][key], state["ground_truth_environment"][key]
                        )
                        if not match:
                            if debug: logger.debug(f"State mismatch found for class: {key}")
                            all_states_match = False
                            break
                    if debug and all_states_match: logger.debug("All environment states match GT.")

            calls_match = False
            try:
                model_calls_raw = state.get("successful_func_calls", [])
                model_calls_parsed = []
                for call in model_calls_raw:
                    normalized_args = {}
                    for k, v in call.get("args", {}).items():
                        normalized_args[k] = tuple(v) if isinstance(v, list) else v
                    model_calls_parsed.append({"name": call.get("name"), "args": normalized_args})

                all_gt_turns_str = state["dataset_row"].get("answer", "[]")
                all_gt_turns = json.loads(all_gt_turns_str)
                current_turn_idx = state["dataset_row"].get("num_turns", 1) - 1

                if current_turn_idx < 0 or current_turn_idx >= len(all_gt_turns):
                    # logger.warning(f"State {i}: Invalid current_turn_idx ({current_turn_idx}) or GT answer length mismatch.")
                    calls_match = not model_calls_parsed
                else:
                    gt_calls_str_for_turn = all_gt_turns[current_turn_idx]
                    gt_calls_parsed = [self._parse_function_call(call_str) for call_str in gt_calls_str_for_turn]
                    gt_calls_parsed = [call for call in gt_calls_parsed if call is not None]

                    # if debug:
                    #     logger.debug(f"Model Calls Parsed: {model_calls_parsed}")
                    #     logger.debug(f"GT Calls Parsed: {gt_calls_parsed}")

                    calls_match = (model_calls_parsed == gt_calls_parsed)
                    if debug: logger.debug(f"Function calls match GT: {calls_match}")

            except json.JSONDecodeError as e:
                # logger.error(f"State {i}: Failed to parse GT answer JSON: {e}")
                calls_match = False
            except Exception as e:
                # logger.error(f"State {i}: Error during function call comparison: {e}")
                calls_match = False

            final_reward = 1.0 if all_states_match and calls_match else 0.0
            rewards.append(final_reward)
            if debug: logger.debug(f"State Match: {all_states_match}, Calls Match: {calls_match} => Final Unified Reward: {final_reward}")
            if debug: logger.debug(f"--- End Unified Reward Check (State {i}) ---")

        return rewards

    def tool_execution_reward_func(
        self,
        completions: List[List[Dict[str, str]]],
        states: List[Dict[str, Any]],
        debug: bool = False,
        max_successful_calls_rewarded: int = 3,
        max_correct_name_calls_rewarded: int = 3,
        reward_per_successful_call: float = 0.05,
        reward_per_correct_name: float = 0.1,
    ) -> List[float]:
        # ... (implementation from previous step) ...
        rewards = []
        for i, state in enumerate(states):
            if debug: logger.debug(f"\n--- Tool Execution Reward Check (State {i}) ---")
            current_reward = 0.0
            successful_calls_count = 0
            correct_tool_name_count = 0

            model_calls = state.get("successful_func_calls", [])
            successful_calls_count = len(model_calls)

            gt_tool_names = set()
            try:
                all_gt_turns_str = state["dataset_row"].get("answer", "[]")
                all_gt_turns = json.loads(all_gt_turns_str)
                current_turn_idx = state["dataset_row"].get("num_turns", 1) - 1

                if 0 <= current_turn_idx < len(all_gt_turns):
                    gt_calls_str_for_turn = all_gt_turns[current_turn_idx]
                    for call_str in gt_calls_str_for_turn:
                        parsed_call = self._parse_function_call(call_str)
                        if parsed_call and "name" in parsed_call:
                            gt_tool_names.add(parsed_call["name"])
                if debug: logger.debug(f"GT Tool Names for Turn {current_turn_idx+1}: {gt_tool_names}")

            except Exception as e:
                # logger.error(f"State {i}: Failed to get GT tool names: {e}")
                pass


            for call in model_calls:
                if call.get("name") in gt_tool_names:
                    correct_tool_name_count += 1

            successful_call_reward = min(successful_calls_count, max_successful_calls_rewarded) * reward_per_successful_call
            correct_name_reward = min(correct_tool_name_count, max_correct_name_calls_rewarded) * reward_per_correct_name

            current_reward = successful_call_reward + correct_name_reward
            rewards.append(current_reward)

            if debug:
                 logger.debug(f"Successful Model Calls: {successful_calls_count}")
                 logger.debug(f"Calls with Correct Name: {correct_tool_name_count}")
                 logger.debug(f"Tool Execution Reward: {current_reward} (Success: {successful_call_reward}, Name: {correct_name_reward})")
            if debug: logger.debug(f"--- End Tool Execution Reward Check (State {i}) ---")

        return rewards

    def format_reward_func(
        self,
        completions: List[List[Dict[str, str]]],
        states: List[Dict[str, Any]],
        debug: bool = False,
        termination_reward: float = 0.1,
        tool_position_reward: float = 0.1,
    ) -> List[float]:
        # ... (implementation from previous step) ...
        rewards = []
        for i, completion in enumerate(completions):
            if not completion or completion[0].get("role") != "assistant":
                #  logger.warning(f"State {i}: No valid assistant completion found for format reward.")
                 rewards.append(0.0)
                 continue

            content = completion[0].get("content", "")
            current_reward = 0.0
            if debug: logger.debug(f"\n--- Format Reward Check (State {i}) ---")
            if debug: logger.debug(f"Content: {content[:50]}...")

            think_end_idx = content.find("</think>")
            task_finished_idx = content.find("<TASK_FINISHED>")
            task_error_idx = content.find("<TASK_ERROR>")

            has_tools = "<tool>" in content
            tools_before_think = False
            if has_tools and think_end_idx != -1:
                last_tool_start_idx = content.rfind("<tool>")
                if last_tool_start_idx != -1 and last_tool_start_idx < think_end_idx:
                    tools_before_think = True
            elif has_tools and think_end_idx == -1:
                pass
            elif not has_tools:
                 tools_before_think = False

            if tools_before_think:
                current_reward += tool_position_reward
                if debug: logger.debug(f"Tool Position Reward: +{tool_position_reward}")

            terminated_correctly = False
            has_termination = task_finished_idx != -1 or task_error_idx != -1
            if has_termination:
                if think_end_idx != -1:
                    if (task_finished_idx != -1 and task_finished_idx > think_end_idx) or \
                       (task_error_idx != -1 and task_error_idx > think_end_idx):
                        terminated_correctly = True

            if terminated_correctly:
                current_reward += termination_reward
                if debug: logger.debug(f"Termination Format Reward: +{termination_reward}")
            elif has_termination and not terminated_correctly:
                if debug: logger.debug("Termination found but format incorrect (missing </think> or wrong order).")

            rewards.append(current_reward)
            if debug: logger.debug(f"Total Format Reward: {current_reward}")
            if debug: logger.debug(f"--- End Format Reward Check (State {i}) ---")

        return rewards

    def self_correction_reward_func(
        self,
        completions: List[List[Dict[str, str]]],
        states: List[Dict[str, Any]],
        debug: bool = False,
        correction_reward: float = 0.1,
    ) -> List[float]:
        # ... (implementation from previous step) ...
        rewards = []
        pattern = re.compile(r"<tool>\s*\{\s*\"name\"\s*:\s*\"(.*?)\".*?\}\s*</tool>\s*<tool_result>(.*?)</tool_result>", re.DOTALL)

        for i, completion in enumerate(completions):
            current_reward = 0.0
            if not completion or completion[0].get("role") != "assistant":
                #  logger.warning(f"State {i}: No valid assistant completion found for self-correction reward.")
                 rewards.append(0.0)
                 continue

            content = completion[0].get("content", "")
            if debug: logger.debug(f"\n--- Self-Correction Reward Check (State {i}) ---")

            tool_attempts: Dict[str, List[bool]] = {}
            correction_found = False

            for match in pattern.finditer(content):
                tool_name = match.group(1)
                result_content = match.group(2).strip()

                is_success = True
                # More robust check for failure indicators
                if result_content.lower().startswith('"error:') or \
                   result_content.lower().startswith('error:') or \
                   "failed" in result_content.lower() or \
                   "error decoding tool call json" in result_content.lower() or \
                   "stray </tool> tag found" in result_content.lower() or \
                   "malformed tool call structure" in result_content.lower():
                    is_success = False


                if tool_name not in tool_attempts:
                    tool_attempts[tool_name] = []
                tool_attempts[tool_name].append(is_success)

                if debug: logger.debug(f"Found tool call: {tool_name}, Success: {is_success}")

            for tool_name, attempts in tool_attempts.items():
                failed_idx = -1
                for idx, success in enumerate(attempts):
                    if not success:
                        failed_idx = idx
                    elif success and failed_idx != -1 and idx > failed_idx:
                        correction_found = True
                        if debug: logger.debug(f"Self-correction detected for tool: {tool_name}")
                        break
                if correction_found:
                    break

            if correction_found:
                current_reward = correction_reward

            rewards.append(current_reward)
            if debug: logger.debug(f"Total Self-Correction Reward: {current_reward}")
            if debug: logger.debug(f"--- End Self-Correction Reward Check (State {i}) ---")

        return rewards

    # --- New Combining Function ---
    def unified_reward_func(
        self,
        completions: List[List[Dict[str, str]]],
        states: List[Dict[str, Any]],
        debug: bool = False,
    ) -> List[float]:
        """
        Calculates the final reward based on the specified logic:
        - If unified success (state + calls match GT) = 1.0:
            final_reward = 1.0 + format_reward
        - If unified success = 0.0:
            final_reward = tool_execution_reward + self_correction_reward + format_reward
        """
        if debug: logger.info("Calculating combined rewards...")

        # Calculate all individual reward components
        unified_rewards = self._reward_components["unified"](completions, states, debug)
        tool_exec_rewards = self._reward_components["tool_exec"](completions, states, debug)
        format_rewards = self._reward_components["format"](completions, states, debug)
        self_correct_rewards = self._reward_components["self_correct"](completions, states, debug)

        final_rewards = []
        num_items = len(states)

        for i in range(num_items):
            unified_r = unified_rewards[i]
            tool_exec_r = tool_exec_rewards[i]
            format_r = format_rewards[i]
            self_correct_r = self_correct_rewards[i]

            final_reward = 0.0
            if unified_r == 1.0:
                # Success case: Base reward is 1.0 + format bonus
                final_reward = unified_r + format_r
                if debug: logger.debug(f"State {i}: Unified Success (1.0) + Format ({format_r}) = {final_reward}")
            else:
                # Failure case: Sum of tool exec, self-correction, and format rewards
                final_reward = tool_exec_r + self_correct_r + format_r
                if debug: logger.debug(f"State {i}: Unified Failure (0.0) -> ToolExec ({tool_exec_r}) + SelfCorrect ({self_correct_r}) + Format ({format_r}) = {final_reward}")

            # Ensure reward is not negative (though current logic shouldn't produce negatives)
            final_rewards.append(max(0.0, final_reward))

        if debug: logger.info("Finished calculating combined rewards.")
        return final_rewards

    def get_reward_funcs(self) -> List[callable]:
        """Returns only the combined reward calculation function."""
        # The trainer will call this single function.
        return [self.unified_reward_func]