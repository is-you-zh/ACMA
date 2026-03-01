reflection_prompt = '''
You are an advanced reasoning agent capable of improving based on self-reflection.  You will be given a previous reasoning trial in which you were provided with access to multiple tools, each with specific functionality, and a question to solve.  
Here is a list of tools and questions you can use:
{functions}
Question:
{question}
You may not be able to answer this question because::
1.You failed to correctly identify and combine the appropriate tools.
2.You have made incorrect assumptions about the parameters of the api.
3.You exceeded your set number of reasoning steps without finding a solution.
4.You provided an incorrect final answer.
Diagnose a possible reason for the failure, considering issues such as incorrect tool selection, suboptimal tool combinations, or insufficient reasoning steps.  Then, devise a new, concise, high-level plan that aims to mitigate the identified failure.  Your plan should outline strategies for better tool selection, efficient API usage, and improved reasoning steps.
Previous trial:
{trajectory}
Reflection:
'''

cot_prompt_feedback = '''
You are an advanced reasoning agent capable of solving complex tasks by using multiple tools(functions) in combination. Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps. 
You can utilize a variety of tools(functions), each serving specific purposes. 
Thought steps allow you to reason about the current situation, while Action steps involve invoking tools(functions) to retrieve or process information. Observation provides the result of the executed Action, which informs subsequent reasoning.
Here are the types of Action you can take:
{tool_des}
After each observation, provide the next Thought and next Action. 
You have attempted to answer the following question before and failed possibly because:
1.You failed to correctly identify and combine the appropriate tools.
2.You made incorrect assumptions about the inputs or outputs of the tools(functions).
3.You exceeded your set number of reasoning steps without finding a solution.
4.You provided an incorrect final answer.

The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.
{trajectories}
When providing the thought and action for the current trial, that into account these failed trajectories and make sure not to repeat the same mistakes and incorrect answers. 
{input}
'''

cot_prompt_feedback_short = '''
You are an advanced reasoning agent capable of solving complex tasks by using multiple APIs in combination. Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps. You can utilize a variety of APIs, each serving specific purposes. Thought steps allow you to reason about the current situation, while Action steps involve invoking APIs to retrieve or process information. Observation provides the result of the executed Action, which informs subsequent reasoning.
If you think you cannot finish the task with the current tools, you can call Finish function to restart and update api pool.
If you have already called an API before, there is no need to call it again.
Here are the types of Action you can take:
tools:
{tool_des}
After each observation, provide the next Thought and next Action. Here are some examples:


You have attempted to answer the following question before and failed because:
1.You failed to correctly identify and combine the appropriate tools.
2.You made incorrect assumptions about the inputs or outputs of the APIs.
3.You exceeded your set number of reasoning steps without finding a solution.
4.You provided an incorrect final answer.

The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.

{trajectories}
When providing the thought and action for the current trial, that into account these failed trajectories and make sure not to repeat the same mistakes and incorrect answers. 

{input}
'''

cot_prompt = '''
You are an advanced reasoning agent capable of solving complex tasks by using multiple apis in combination. Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps.
You can use many apis to do the following task. Note the description of each tool. Here are the apis you can use:
{functions}
After each observation, offer the next Thought and the next Action.
Thought can reason about the current situation and think about the next action, while Action steps involve invoking apis to retrieve or process information, Observation provides the result of the executed Action.
Note: You can only use one api at a time, output only the name of the api, if there are parameters, you can output api_name{{parameter}}
Remember, if you have used an API before, do not call it again with the same parameters, because the result of your repeated call will still be the same. You can try other APIs.
If the current tools cannot complete the task, call Finish function to restart and give the reason.
trajectory:
{input}
'''

cot_prompt_with_tool = '''
You are an advanced reasoning agent capable of solving complex tasks by using multiple apis in combination. Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps. If the tool doesn't solve the problem, you can try to answer it based on your own knowledge.
You can use many apis to do the following task. Note the description of each api.
After each observation, **offer the next Thought and the next Action**.
Thought can reason about the current situation and think about the next action, while Action steps involve invoking apis to retrieve or process information, Observation provides the result of the executed Action.

Such as:
Thought 1: To provide the travel blogger with the necessary currency conversion rates, I need to first identify the list of supported currencies and then fetch the current exchange rates from USD to EUR, GBP, and AUD. 
Action 1: currencies_for_currencyapi_net{{"cuntry":"us"}}

Important:
1. **Use only one api at a time**, output only the name of the api, if there are parameters, you can output api_name{{parameter}}
The parameters need to be in dictionary format, for example, {{"location":"beijing"}}
When **calling 'Finish', parameters must be provided**. Parameter examples:{{"return_type": "give_answer", "final_answer": "The current price of gold is $1740.93 per ounce. Please provide the symbol of the stock you are interested in to continue gathering the financial data and earnings estimates."}}
2. If you have already called an API with specific parameters, avoid calling it again with the same ones, as the result will not change.
3. If the tool does not return a direct answer, but clearly indicates that the tool can solve the user's problem, or it requires further actions from the user, this part of the task can also be considered completed.
4. If you can solve part of the problem with your own knowledge, then that part doesn't require a tool call.
5. If the current tools cannot complete the task, call Finish function to restart or give up and give the reason, or return answer based on your knowledge. Don't fabricate tools.

trajectory:
{input}
'''

cot_prompt_short = '''
Your task is to solve a given question by reasoning through interleaved Thought, Action, and Observation steps. You can utilize a variety of APIs, each serving specific purposes. Thought steps allow you to reason about the current situation, while Action steps involve invoking APIs to retrieve or process information. Observation provides the result of the executed Action, which informs subsequent reasoning.
Here are the types of Action you can take:
tools:
{tool_des}

{input}
'''

value_prompt_reasoning = '''
You are an advanced reasoning agent that can improve based on self refection. Analyze the trajectories of your previous solutions to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions.
Given a question and a trajectory, evaluate its correctness and you don't need to provide reasoning and analysis. Pay particular attention to the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Finally, you only need to output a number for the correctness score, which is an integer between 1 and 10.
{input}
'''

value_prompt_reasoning_together = '''
You are an advanced reasoning agent capable of evaluating strategies based on past experiences. Analyze the trajectories of your previous solutions to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions.
Given a question and multiple trajectories, evaluate its correctness and you don't need to provide reasoning and analysis. Pay particular attention to the latest thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. 
Finally, you only need to output a set of numbers for the correctness score, which are some integers between 1 and 10. Such as[10, 9, 10, 7, 3]
Output as many numbers as there are child nodes.
{input}
'''

CHECK_SOLVABLE_BY_FUNCTION_PROMPT_FOR_EXPERIENCE = """
Your responsibility is to assess whether the given `available_tools` can handle the `query`. 
Please evaluate with these three possible results:

1. If the available APIs, based on their names and descriptions, provide all the necessary functions to complete the task, the query can be considered fully solvable, return:
{
    "solvable": "FullySolvable",
    "reason": "Explain why the tools are sufficient."
}

2. If the available tools can partially resolve the query, return:
{
    "solvable": "PartiallySolvable",
    "reason": "Explain which parts can be covered and which parts are not covered.",
    "uncovered_subqueries": "Write **specific, human-readable subqueries** for the uncovered parts. For example: 'Get the timezone information for Los Angeles.'"
}

3. If the available tools cannot provide a solution to the query, return:
{
    "solvable": "ToolsNotAvailable",
    "reason": "Explain why no current tools are applicable."
}

Important:
- Uncovered_subqueries should **merge all unmet needs into a single, clear natural language query**.
- Never omit any unmet requirement from the original query.
- If a tool's functional category covers the user's need and no explicit limitations are stated, assume the tool is capable of fulfilling that need.

Here are some examples:
---
Example 1:
{
    "solvable": "FullySolvable",
    "reason": "The tools can fully handle the request to get weather information and restaurant recommendations in Paris."
}

Example 2:
{
    "solvable": "PartiallySolvable",
    "reason": "The tools can provide restaurant recommendations but cannot fetch the current weather information.",
    "uncovered_subqueries": "Get the current weather in Paris."
}

Example 3:
{
    "solvable": "ToolsNotAvailable",
    "reason": "The query involves translating an ancient script, and no tools are available for that task."
}
"""

RESUME_PROMPT = """
You are an intelligent assistant. Based on the failure reason provided by the execution agent, your task is to generate a new set of subtasks to resolve the query.

Please strictly follow the output format (JSON):
{
  "task_list": [[subtask1, sublabel1], [subtask2, sublabel2], [subtask3, sublabel3] (if any)]
}

**Important rules to follow:**
1. Refer to the failure reason provided by the execution agent to identify what caused the failure. After reviewing the failure reason and candidate APIs, generate a new list of **explicit and actionable subtasks** to help resolve the query. Each subtask should correspond to a clear action or information the user needs.
2. Ensure the subtasks are **specific and actionable** without including parameters such as names.
3. Output must be in valid JSON format. **Do NOT** include extra explanations or comments.

Here is an example of how you should proceed:

[Input]: The request failed because the current tool could not retrieve real-time weather data from the specified city. We have a pool of candidate APIs that may handle weather queries.

[Output]:
{
  "task_list": [
    {
      "subtask": "Retrieve real-time weather data for the specified city",
      "sublabel": "Weather Data"
    },
    {
      "subtask": "Use backup weather API to fetch current weather",
      "sublabel": "Weather Data"
    }
  ]
}

Now, based on the failure reason and the available candidate APIs, decompose the following query into subtasks that cannot be resolved by the current tools are:
Query: {query} The retrieved API: {api_pool} Fail reason: {fail_reason} Deleted API:{delete_api}
"""

INTENT_DECOMPOSER_PROMPT = """
You are an intelligent assistant. Your task is to understand the user's request and transform it into a structured output containing a **task list (task_list)** and a **task scene label (task_scene)**.

Please strictly follow the output format (JSON):
{
  "task_list": [[subtask1, sublabel1], [subtask2, sublabel2], [subtask3, sublabel3] (if any)],
  "task_scene": "Task scene label"
}

The goal is to help the system identify the user's **core intentions** and extract clear subtasks. Each subtask represents a specific type of information or action the user explicitly requests, so the system can reuse this in future experience retrieval.

**Important rules to follow:**
1. Only extract **explicitly requested** subtasks. Ignore background details. **Do NOT** add any unrequested content, even if it seems reasonable.
2. Each subtask should be **specific and actionable**. Avoid vague or general terms (for example, replace “check transportation” with “check train tickets” or “check flight details”).
3. Connect subtasks using “+”. Keep the granularity consistent and the expressions clear and unambiguous.
4. **Do NOT** include any **parameters** (such as location names, person names, dates, etc.) in the subtask descriptions. Focus only on the type of action or information the user wants.
5. Output must be in valid JSON format. **Do NOT** include extra explanations or comments.

Below are reference examples:

[Input]: Please provide the CO2 emissions for electricity in Germany and all recent real estate transactions for the 10001 zip code. I'm researching how energy consumption correlates with property activities in that region.
[Output]:
{
  "task_list": [
    {
      "subtask": "Retrieve CO2 emissions data",
      "sublabel": "Environmental Data"
    },
    {
      "subtask": "Obtain real estate transaction data",
      "sublabel": "Market Data"
    }
  ],
  "task_scene": "Environmental and Market Research"
}


[Input]: Please tell me the latest financial report of Company A and its shareholder structure.
[Output]:
{
  "task_list": [
    {
      "subtask": "Retrieve Company A's latest financial report",
      "sublabel": "Financial Data"
    },
    {
      "subtask": "Obtain Company A's shareholder structure",
      "sublabel": "Ownership Data"
    }
  ],
  "task_scene": "Corporate Financial and Ownership Analysis"
}


[Input]: I want to fly from Paris (Charles de Gaulle Airport) to Tokyo (Haneda Airport). Please check the terminal information for both airports, recommend 3 hotels near Haneda, and tell me the next available train from Haneda to Ginza Station.
[Output]:
{
  "task_list": [
    {
      "subtask": "Check terminal info for CDG and HND",
      "sublabel": "Airport Info"
    },
    {
      "subtask": "Recommend 3 hotels near HND",
      "sublabel": "Hotel Info"
    },
    {
      "subtask": "Find next train from HND to Ginza",
      "sublabel": "Transport Info"
    }
  ],
  "task_scene": "Travel Info"
}


Now, please decompose the following user request into structured intentions and generate the output:
[Input]: {{USER_INPUT}}
[Output]:
"""

FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION = """
You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: {task_description}"""

FORMAT_INSTRUCTIONS_USER_FUNCTION = """
{input_description}
Begin!
"""

FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ADAPTED = """
You are AutoGPT, you can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.
After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer. 
If you feel you cannot solve the task or can only solve it partially, you should choose to give up and give your reason which should mention the names of the failed functions.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart" and give the reason.
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.
Let's Begin!
Task description: {task_description}"""

DIVERSITY_PROMPT='''
This is not the first time you try this task, all previous trails failed.
Before you generate my thought for this state, I will first show you your previous actions for this state, and then you must generate actions that is different from all of them. Here are some previous actions candidates:
{previous_candidate}
Remember you are now in the intermediate state of a trail, you will first analyze the now state and previous action candidates, then make actions that is different from all the previous.'''

LLM_PAIRWISE_RANK_SUBFIX_SYSTEM_PROMPT = '''
You are value-GPT, which is an expert of defining which trail is better, which trail is more close to solving the task. 
All candidate tries to solve this task with some funciton calls:
*******************************
{{TASK_DESCRIPTION}}
{task_description}
{{END_TASK_DESCRIPTION}}
*******************************
First, all candidate do the following things:
{intersect_trice}
After that, there are two candidates A and B, they do different things:
*******************************
{{CANDIDATE_A_START}}
{candidate_A}
{{CANDIDATE_A_END}}
*******************************
{{CANDIDATE_B_START}}
{candidate_B}
{{CANDIDATE_B_END}}
Which try do you think is more helpful to solving the task?
'''

LLM_PAIRWISE_RANK_USER_PROMPT = '''
Tell me which candidate is better in ONE Word: "A" or "B":'''

RETRIEVAL_PROMPT = """
You are RetrievalAgent, a specialized agent designed to retrieve and assign APIs to solve tasks. You have access to a database of APIs. Your task is to execute the necessary steps to complete a given task using the relevant tools. You can use the following tools:

1. **decompose_task**: if necessary, use this function to break down the task description into smaller, independent sub-tasks that can be handled individually.

2. **search_apis**: this function to retrieve the most relevant tools for each sub-task from the cache or tool library. Do not repeatedly execute the search_apis with exactly the same parameters.

3. **check_task_completion**: this function is used to check whether all sub-tasks can be completed by the assigned tools. If any sub-task is not completed, then the sub-task should be further broken down or additional tools should be obtained.

4. **finish_func**: when you feel that the task has been completed or is impossible to complete and get stuck in a loop, call this function.
Your goal is to efficiently allocate the right tools to the right sub-tasks and ensure that the overall task is completed successfully.
Important: Do not repeat performing actions with the same parameters.
"""

CHECK_TASK_COMPLETION_PROMPT = """
You are a Tool Validator. Your task is to analyze whether the currently available tools are sufficient to complete the specified user task.

### Input
1. **User Task**: {task}
2. **Available Tools**: {tools}

### Output Requirement
You must return a **single Python dictionary** with exactly two keys:
1. `solvable` (bool): True if the tools are sufficient, False otherwise.
2. `reason` (str): A brief explanation of why.

**Output Example:**
{
    "solvable": false,
    "reason": "The tool pool includes 'search_box_office' to check rankings, but lacks a ticket purchasing tool (such as 'buy_ticket' or 'book_cinema_ticket') to complete the transaction."
}

{
    "solvable": true,
    "reason": "The available 'get_weather' tool can retrieve the weather forecast for Beijing tomorrow."
}

"""

EVALUATE_COMPLETION_SCORE_PROMPT = """
You are an expert evaluator assessing how well a system response fulfills a user query. Based on the following criteria, output only a numeric score between 0.00 and 1.00 (e.g., 0.75). Do not provide any explanation.

[Evaluation Criteria]:
· 0.00: Completely irrelevant or useless response; fails to address the query.
- 0.10-0.69: Partially relevant; some key elements are addressed, but there are major omissions, logical errors, or broken steps. Clearly indicate that it has been resolved, but no detailed answer is given.
- 0.70-0.99: Mostly complete; the main task is addressed, with only minor issues in detail, completeness, or precision.
- 1.00: Fully complete, accurate, and logically sound; no significant issues, no further clarification needed.

Notes:
- Ignore irrelevant or redundant content returned by the system (e.g., boilerplate or verbose API output).
- The primary evaluation criterion is the degree of task completion.

[Example]:
Query: Please provide the auditor details for ROAC number 87654321 and also return the company financial summary for company ID 54321. What is the temperature in Wuhan today?
Score 1.00
Response:
"Auditor info: Name: Carmen Vázquez Peña, ROAC: 87654321, Document: 33345107Y. Company financial summary for ID 54321: Revenue: $5M, Profit: $1M, Fiscal Year: 2023. Note: API version used is v1.17.0. The temperature in Wuhan is 35°C today."

Score 0.86
Response:
"Auditor info: Name: Carmen Vázquez Peña, ROAC: 87654321, Document: 33345107Y. Company financial summary for ID 54321: Revenue: $5M, Profit: $1M, Fiscal Year: 2023."

Score 0.65
Response:
"Auditor info: Name: Carmen Vázquez Peña. Company financial summary not available. Wuhan temp: 35°C."

Score 0.30
Response:
"The ROAC 87654321 is related to an auditor in Spain. Company financial data might be confidential."

Score 0.00
Response:
"Cats are great pets and love to jump around."

[User Query]:
{query}

[System Response]:
{system_response}
"""

COMPLETE_SUBTASK_PROMPT = """
You are an intelligent data annotation assistant. Your task is to extract called tools based on the execution trajectory and fill their complete information objects into the subtasks.
Input Data
subtasks: The target list to be filled.
execution_strategy: The execution trajectory containing actual call records.
tool_details: The standard tool detail list (containing category_name, tool_name, api_name).
Processing Logic
1. Extract the called function names from execution_strategy.tool_order (e.g., functions.get_equations_for_...). Ignore items starting with Question and ending with Finish.
2. Based on keywords in the function name (e.g., get_equations), find the matching entry in the tool_details list.
3. Fill the complete JSON object of that entry into the api field of the semantically most appropriate subtasks. If one intent corresponds to multiple tools, the api field should be a list containing multiple objects.
Note: The api field should store an object (or a list of objects), not a string. Finally, only the subtasks field is returned.
Example: 
{
  "subtasks": [
    {
      "intent": "Find appetizer recipe with avocado",
      "api": {
        "category_name": "Food",
        "tool_name": "RecipeFinder",
        "api_name": "appetizer/ingredient"
      }
    }
  ]
}
Input: {experience}
"""

ALIASES_PROMPT = """
You are a data augmentation and semantic expansion expert. Your task is to generate an aliases field based on the provided “intent” and “API description”.

Requirements:
Analyze: Carefully read the provided intent description and the corresponding API function descriptions.
Generate: Create a list of three synonymous paraphrases for the "intent".
Purpose: These aliases should assist the system in fuzzy matching, covering different expressions a user might use (such as synonyms, abbreviations, or colloquial expressions).
Format: The output must be a valid Python list, e.g. ["alias1", "alias2", "alias3"].
Input: {input}
"""
