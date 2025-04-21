# bfcl_inthinking_rubric.py

import ast
import json
import re
import time
from typing import Any, Dict, List, Tuple # Added Tuple

from loguru import logger

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric


class BfclITRubric(Rubric):
    def __init__(
        self,
        parser: XMLParser = XMLParser(fields=["think", "tool"]),
        env_parser: XMLParser = XMLParser(fields=["tool_result"]),
        # Define reward values here for easier tuning
        termination_reward_val: float = 0.1,
        tool_position_reward_val: float = 0.1,
        reward_per_successful_call_val: float = 0.05,
        reward_per_correct_name_val: float = 0.1,
        correction_reward_val: float = 0.1,
        max_successful_calls_rewarded_val: int = 3,
        max_correct_name_calls_rewarded_val: int = 3,
    ):
        self.parser = parser
        self.env_parser = env_parser
        # Store reward values
        self.termination_reward_val = termination_reward_val
        self.tool_position_reward_val = tool_position_reward_val
        self.reward_per_successful_call_val = reward_per_successful_call_val
        self.reward_per_correct_name_val = reward_per_correct_name_val
        self.correction_reward_val = correction_reward_val
        self.max_successful_calls_rewarded_val = max_successful_calls_rewarded_val
        self.max_correct_name_calls_rewarded_val = max_correct_name_calls_rewarded_val

        # Store individual component functions internally
        self._reward_components = {
            "unified": self.unified_success_reward_func,
            "tool_exec": self.tool_execution_reward_func,
            "format": self.format_reward_func,
            "self_correct": self.self_correction_reward_func,
        }

    @staticmethod
    def _parse_function_call(func_call_str: str) -> Dict | None:
        # ... (implementation unchanged) ...
        try:
            tree = ast.parse(func_call_str.strip(), mode='eval')
            if not isinstance(tree.body, ast.Call): return None
            func_node = tree.body.func
            if isinstance(func_node, ast.Name): func_name = func_node.id
            elif isinstance(func_node, ast.Attribute): func_name = func_node.attr
            else: return None
            args_dict = {}
            for kw in tree.body.keywords:
                try:
                    value = ast.literal_eval(kw.value)
                    args_dict[kw.arg] = value
                except ValueError: pass
                except Exception as e: logger.warning(f"Error evaluating keyword arg '{kw.arg}' in '{func_call_str}': {e}")
            for i, arg in enumerate(tree.body.args):
                try: args_dict[f"pos_arg_{i}"] = ast.literal_eval(arg)
                except ValueError: pass
                except Exception as e: logger.warning(f"Error evaluating positional arg {i} in '{func_call_str}': {e}")
            for key, value in args_dict.items():
                if isinstance(value, list): args_dict[key] = tuple(value)
            return {"name": func_name, "args": args_dict}
        except Exception as e: return None

    @staticmethod
    def compare_instances(model_object, ground_truth_object):
        # ... (implementation unchanged) ...
        if type(model_object) != type(ground_truth_object): return False
        match = True
        for attr_name in vars(ground_truth_object):
            if attr_name.startswith("_"): continue
            if not hasattr(model_object, attr_name): match = False; break
            model_attr = getattr(model_object, attr_name)
            ground_truth_attr = getattr(ground_truth_object, attr_name)
            if model_attr != ground_truth_attr: match = False; break
        return match

    # --- MODIFIED Signature ---
    def unified_success_reward_func(
        self,
        completions: List[str], # Changed type hint
        states: List[Dict[str, Any]],
        debug: bool = False,
    ) -> List[float]:
        rewards = []
        for i, state in enumerate(states):
            # No need to check completion structure, assume completions[i] is the string
            completion_content = completions[i] # Use the string directly
            if not completion_content: # Check if string is empty
                 logger.warning(f"State {i}: Empty completion string found for unified reward.")
                 rewards.append(0.0)
                 continue

            if debug: logger.debug(f"\n--- Unified Reward Check (State {i}) ---")

            # State comparison logic remains the same
            all_states_match = True
            if "ground_truth_environment" not in state or "environment" not in state:
                all_states_match = False
            else:
                if state["ground_truth_environment"].keys() != state["environment"].keys():
                     all_states_match = False
                else:
                    for key in state["ground_truth_environment"]:
                        if key not in state["environment"]:
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

            # Function call comparison logic remains the same (uses state)
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
                    calls_match = not model_calls_parsed
                else:
                    gt_calls_str_for_turn = all_gt_turns[current_turn_idx]
                    gt_calls_parsed = [self._parse_function_call(call_str) for call_str in gt_calls_str_for_turn]
                    gt_calls_parsed = [call for call in gt_calls_parsed if call is not None]
                    calls_match = (model_calls_parsed == gt_calls_parsed)
                    if debug: logger.debug(f"Function calls match GT: {calls_match}")
            except json.JSONDecodeError as e: calls_match = False
            except Exception as e: calls_match = False

            final_reward = 1.0 if all_states_match and calls_match else 0.0
            rewards.append(final_reward)
            if debug: logger.debug(f"State Match: {all_states_match}, Calls Match: {calls_match} => Final Unified Reward: {final_reward}")
            if debug: logger.debug(f"--- End Unified Reward Check (State {i}) ---")

        return rewards

    # --- MODIFIED Signature ---
    def tool_execution_reward_func(
        self,
        completions: List[str], # Changed type hint
        states: List[Dict[str, Any]],
        debug: bool = False,
    ) -> List[float]:
        # This function primarily uses 'states', so internal logic is unchanged
        # Only the signature needs updating for consistency.
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
            except Exception as e: pass
            for call in model_calls:
                if call.get("name") in gt_tool_names:
                    correct_tool_name_count += 1
            successful_call_reward = min(successful_calls_count, self.max_successful_calls_rewarded_val) * self.reward_per_successful_call_val
            correct_name_reward = min(correct_tool_name_count, self.max_correct_name_calls_rewarded_val) * self.reward_per_correct_name_val
            current_reward = successful_call_reward + correct_name_reward
            rewards.append(current_reward)
            if debug:
                 logger.debug(f"Successful Model Calls: {successful_calls_count}")
                 logger.debug(f"Calls with Correct Name: {correct_tool_name_count}")
                 logger.debug(f"Tool Execution Reward: {current_reward} (Success: {successful_call_reward}, Name: {correct_name_reward})")
            if debug: logger.debug(f"--- End Tool Execution Reward Check (State {i}) ---")
        return rewards

    # --- MODIFIED Signature & Logic ---
    def format_reward_func(
        self,
        completions: List[str], # Changed type hint
        states: List[Dict[str, Any]],
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        results = []
        for i, completion_content in enumerate(completions): # Iterate directly over strings
            tool_pos_score = 0.0
            term_score = 0.0
            term_tag_used = "NONE"

            # Use completion_content directly
            content = completion_content
            if not content: # Handle empty string case
                 logger.warning(f"State {i}: Empty completion string found for format reward.")
                 results.append({'tool_pos': 0.0, 'term': 0.0, 'term_tag': 'NONE'})
                 continue

            if debug: logger.debug(f"\n--- Format Reward Check (State {i}) ---")
            if debug: logger.debug(f"Content: {content[:50]}...") # Use content directly

            think_end_idx = content.find("</think>")
            task_finished_idx = content.find("<TASK_FINISHED>")
            task_error_idx = content.find("<TASK_ERROR>")
            # Ignore TASK_ERROR if added by env due to max steps
            if task_error_idx != -1 and "Max tool interactions reached." in content[task_error_idx:]:
                task_error_idx = -1
            elif task_error_idx != -1 and "Max steps reached." in content[task_error_idx:]:
                 task_error_idx = -1
            elif task_error_idx != -1 and "Max tokens reached." in content[task_error_idx:]:
                task_error_idx = -1

            # Check tool position
            has_tools = "<tool>" in content
            tools_before_think = False
            if has_tools and think_end_idx != -1:
                last_tool_start_idx = content.rfind("<tool>")
                last_tool_end_idx = content.rfind("</tool>")
                if last_tool_start_idx != -1 and last_tool_end_idx != -1 and last_tool_end_idx < think_end_idx:
                    tools_before_think = True
            elif not has_tools:
                 tools_before_think = False # No tools, position is correct

            if tools_before_think:
                tool_pos_score = self.tool_position_reward_val
                if debug: logger.debug(f"Tool Position Reward: +{tool_pos_score}")

            # Check termination format and identify tag
            has_finish_tag = task_finished_idx != -1
            has_error_tag = task_error_idx != -1
            has_termination = has_finish_tag or has_error_tag

            if has_termination:
                if think_end_idx != -1 and \
                   ((has_finish_tag and task_finished_idx > think_end_idx) or \
                    (has_error_tag and task_error_idx > think_end_idx)):
                    term_score = self.termination_reward_val
                    term_tag_used = "FINISHED" if has_finish_tag else "ERROR"
                    if debug: logger.debug(f"Termination Format Correct ({term_tag_used}): +{term_score}")
                # else:
                    # if debug: logger.debug("Termination found but format incorrect (missing </think> or wrong order).")

            results.append({
                'tool_pos': tool_pos_score,
                'term': term_score,
                'term_tag': term_tag_used
            })
            if debug: logger.debug(f"Format Result: {results[-1]}")
            if debug: logger.debug(f"--- End Format Reward Check (State {i}) ---")

        return results

    # --- MODIFIED Signature & Logic ---
    def self_correction_reward_func(
        self,
        completions: List[str], # Changed type hint
        states: List[Dict[str, Any]],
        debug: bool = False,
    ) -> List[float]:
        rewards = []
        pattern = re.compile(r"<tool>\s*\{\s*\"name\"\s*:\s*\"(.*?)\".*?\}\s*</tool>\s*<tool_result>(.*?)</tool_result>", re.DOTALL)

        for i, completion_content in enumerate(completions): # Iterate directly over strings
            current_reward = 0.0
            # Use completion_content directly
            content = completion_content
            if not content: # Handle empty string case
                 logger.warning(f"State {i}: Empty completion string found for self-correction reward.")
                 rewards.append(0.0)
                 continue

            if debug: logger.debug(f"\n--- Self-Correction Reward Check (State {i}) ---")

            tool_attempts: Dict[str, List[bool]] = {}
            correction_found = False

            for match in pattern.finditer(content): # Use content directly
                tool_name = match.group(1)
                result_content = match.group(2).strip()
                is_success = True
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
                    if not success: failed_idx = idx
                    elif success and failed_idx != -1 and idx > failed_idx:
                        correction_found = True
                        if debug: logger.debug(f"Self-correction detected for tool: {tool_name}")
                        break
                if correction_found: break

            if correction_found:
                current_reward = self.correction_reward_val

            rewards.append(current_reward)
            if debug: logger.debug(f"Total Self-Correction Reward: {current_reward}")
            if debug: logger.debug(f"--- End Self-Correction Reward Check (State {i}) ---")

        return rewards

    # --- MODIFIED Signature ---
    def calculate_combined_reward(
        self,
        completions: List[str], # Changed type hint
        states: List[Dict[str, Any]],
        debug: bool = False,
    ) -> List[float]:
        if debug: logger.info("Calculating combined rewards...")

        # Calls will now correctly receive List[str] for completions
        unified_rewards = self._reward_components["unified"](completions, states, debug)
        tool_exec_rewards = self._reward_components["tool_exec"](completions, states, debug)
        format_rewards_detailed = self._reward_components["format"](completions, states, debug)
        self_correct_rewards = self._reward_components["self_correct"](completions, states, debug)

        final_rewards = []
        num_items = len(states)

        for i in range(num_items):
            unified_r = unified_rewards[i]
            tool_exec_r = tool_exec_rewards[i]
            format_info = format_rewards_detailed[i]
            self_correct_r = self_correct_rewards[i]

            final_reward = 0.0
            final_reward += format_info['tool_pos'] # Add tool position reward first

            if unified_r == 1.0:
                final_reward += 1.0 # Add base success reward
                if format_info['term_tag'] == 'FINISHED':
                    final_reward += format_info['term'] # Add termination bonus for FINISHED
                if debug: logger.debug(f"State {i}: Unified Success (1.0). Base Format (tool_pos={format_info['tool_pos']}, term={format_info['term'] if format_info['term_tag'] == 'FINISHED' else 0.0}) -> Final: {final_reward}")
            else:
                # Add failure-case rewards
                final_reward += tool_exec_r + self_correct_r
                if format_info['term_tag'] == 'ERROR':
                    final_reward += format_info['term'] # Add termination bonus for ERROR
                if debug: logger.debug(f"State {i}: Unified Failure (0.0). ToolExec ({tool_exec_r})+ SelfCorrect ({self_correct_r}) + Base Format (tool_pos={format_info['tool_pos']}, term={format_info['term'] if format_info['term_tag'] == 'ERROR' else 0.0}) -> Final: {final_reward}")

            final_rewards.append(max(0.0, final_reward))

        if debug: logger.info(f"Finished calculating combined rewards. Final Rewards: {final_rewards}")
        return final_rewards

    def get_reward_funcs(self) -> List[callable]:
        """Returns only the combined reward calculation function."""
        return [self.calculate_combined_reward]