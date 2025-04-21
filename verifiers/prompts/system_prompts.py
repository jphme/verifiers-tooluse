SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

CODE_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

For each step:
1. Think through your reasoning inside <reasoning> tags
2. Write Python scripts inside <code> tags to work out calculations
   - Functions and variables do not persist across <code> calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
3. You will see the output from print() statements in your code in <output> tags
4. Continue until you can give the final answer inside <answer> tags
"""

DEFAULT_TOOL_PROMPT_TEMPLATE = """\
You have access to the following tools to help solve problems:

{tool_descriptions}

For each step:
1. Think through your reasoning inside <reasoning> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Continue until you can give the final answer inside <answer> tags

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""

# BFCL_PROMPT = """\
# You are an expert in composing functions. You are given a question from a user and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to complete the task.
# You have access to the following tools to help solve the task:

# {tools}

# For each step:
# 1. Start with a step-by-step thinking process inside <reasoning> </reasoning> tags. You must provide reasoning in the beginning of your response.
# 2. If needed, use tools by writing one or more JSON commands as a list inside <tool> </tool> tags. Each item in the list should have a name and args key, with args being a dictionary.
#    example: <tool> [{{"name": "tool_name_1", "args": {{"arg1": "value1", "arg2": "value2"}}}}, {{"name": "tool_name_2", "args": {{"arg3": "value3", "arg4": "value4"}}}}] </tool>
# 3. After you have used the tools, you will see the tool outputs inside <tool_result> </tool_result> tags in the same order from the system.
# 4. After calling the tools and seeing the tool results, if you believe the current task is complete, include the <TASK_FINISHED> tag in the end of your response.
# 5. If the task is not able to be completed with the given tools, include the <TASK_ERROR> tag in the end of your response.

# Tools expect specific JSON input formats. Do not make up tools or arguments that aren't listed.
# """

BFCL_PROMPT = """\
You are an expert in composing functions. You are given a question from a user and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to complete the task.
You have access to the following tools to help solve the task:

{tools}

For each step:
1. Start with a step-by-step thinking process inside <reasoning> </reasoning> tags to think through the problem.
2. If needed, use tools by writing one or more JSON commands as a list inside <tool> </tool> tags. Each item in the list should have a name and args key, with args being a dictionary.
   example: <tool> [{{"name": func_1_name, "args": {{arg1: value1, arg2: value2}}}}, {{"name": func_2_name, "args": {{arg3: value3, arg4: value4}}}}] </tool>
   Tools expect specific JSON input formats. Do not make up tools or arguments that aren't listed.
3. After you have used the tools, you will see the tool outputs inside <tool_result> </tool_result> tags in the same order from the system.
4. If you believe the current task is completed and no more tool, summarize your progresses and output <TASK_FINISHED> in the end of your response to terminate the conversation.
5. Otherwise if you believe the task is not able to be completed, summarize what is problematic and output <TASK_ERROR> in the end of your response to terminate the conversation.
"""
