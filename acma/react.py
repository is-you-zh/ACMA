import json
import re
from openai import OpenAI
from config import *
import toolenv
import logging
from arguments import parse_args
from utils import *
from long_term_memory import ExecutionStrategy, Metadata, ToolDetail
import time
from models import call_gpt

client = OpenAI(api_key=api_key, base_url=base_url)

def build_prompt(api_docs_text, scratchpad, user_input):
    prompt = f"""You are a helpful assistant. You can use the following tools:
    {api_docs_text}

    Use the following format:

    Question: the input question you must answer
    Thought: think about what to do next
    Action: the action to take, should be a tool name
    Action Input: a JSON object of tool arguments
    Observation: the result of the action
    ... (this Thought/Action/Observation can repeat N times)
    Thought: I now know the final answer
    Action: Finish
    Action Input: {{
    "return_type": "give_answer",
    "final_answer": "..."
    }}

    Important:
    - When you think you have finished after calling a tool, please call the 'Finish' tool in the next round.
    - Only ONE tool call per turn, **`Finish` also is a tool, it cannot be called simultaneously with other tools**.
    - You MUST call `Finish` tool to end, using one of three `return_type` values:
    - "give_answer": you've completed the task, return answer
    - "give_up_and_restart": current tools failed, restart
    - "give_up": task cannot be solved
    - Do NOT end with free text. Always use Finish.
    Reminder to ALWAYS respond with a single action. Use tools if necessary. Respond directly if appropriate. 
    Begin!

    Question: {user_input}
    {scratchpad}"""
    return [{"role": "system", "content": prompt.strip()}]

def add_experience(experience, scratchpad, query_id, env, api_list, query):
    tool_order, used_api = get_action_trajectory_from_text(scratchpad, query)
    experience.execution_strategy = ExecutionStrategy(
        type="sequential",
        merge_logic="merge",
        tool_order=tool_order
    )
    timestamp = time.time()
    readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    experience.metadata = Metadata(
        created_by="admin",
        created_at=str(readable_time),
        reuse_count=0,
        source_task_id=query_id
    )
    selected_api_list = []
    for item in used_api:
        if item in env.functions_names:
            idx = env.functions_names.index(item)
            selected_api_list.append(api_list[idx])
    experience.tool_details = [
        ToolDetail(**{k: v for k, v in d.items() if k != 'description'}) 
        for d in selected_api_list
    ]
    # print("experience:",experience)
    # add_experience_to_db(experience)
    logger.info(f"[INFO] The experience was added successfully: {experience}")
    return experience

def get_action_trajectory_from_text(text, query):
    trajectory = []
    func_names = []

    trajectory.append(f"User Query: {query}")
    pattern = re.compile(
        r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(\{.*?\})",
        re.DOTALL
    )

    for i, match in enumerate(pattern.finditer(text)):
        func_name = match.group(1).strip()
        raw_input = match.group(2).strip()

        try:
            parsed_input = json.loads(raw_input)
        except json.JSONDecodeError:
            try:
                fixed = raw_input.replace("'", '"').replace("True", "true").replace("False", "false")
                parsed_input = json.loads(fixed)
            except:
                parsed_input = raw_input  

        action_str = f"{func_name}({json.dumps(parsed_input, ensure_ascii=False)})"
        trajectory.append(f"Action {i+1}: {action_str}")
        func_names.append(func_name)
    return trajectory, func_names

def react_search(args, query_data, api_list, i, tool_des, experience=None):
    query = query_data['query']
    env = toolenv.ToolEnv(query=query, api_list=api_list, tool_des=tool_des, process_id=i)
    global logger
    logger = logging.getLogger(f'task_{query_data["query_id"]}')
    print(f"Now playing {query}, with {len(api_list)} APIs")
    logger.info(f"Now playing {query}, with {len(api_list)} APIs")
    solve_tokens = 0
    scratchpad = ""
    final_exp = None

    messages = build_prompt(env.functions, scratchpad, query)
    logger.info(f"Prompt: {messages}")
    finish_hit = False
    for turn in range(10):    
        print(f"\n=== the {turn + 1}-th turn ===")
        logger.info(f"the {turn + 1}-th turn")
        messages = build_prompt(env.functions, scratchpad, query)
        response = call_gpt(model=model_name, messages=messages)
        solve_tokens += response.usage.total_tokens
        if response.choices[0].message.content:
            output = response.choices[0].message.content.strip()
        print("🔁 LLM 输出:\n", output)
        logger.info(f"output: {output}")

        pattern = r"Action:\s*(\w+)\s*Action Input:\s*({(?:[^{}]|(?:\{[^{}]*\}))*})"
        matches = re.findall(pattern, output)

        actions_list = []
        for action, input_str in matches:
            try:
                parsed_input = json.loads(input_str)
            except Exception as e:
                print("❌ JSON 解析失败:", e)
                logger.info(f"JSON 解析失败: {e}")
                continue
            actions_list.append({
                "action": action.strip(),
                "input": parsed_input
            })

        for a in actions_list:
            try:
                if a["action"] == "Finish":
                    print("===========================================")
                    scratchpad += f"{output}"
                    return_type = a["input"].get("return_type")

                    if return_type == "give_answer":
                        print("🎯 最终回答:", a["input"].get("final_answer", "（无内容）"))
                        logger.info(f"Final Answer: {a['input'].get('final_answer', '（无内容）')}")
                        # output = str(a["input"])
                        # output = f"Action: Finish Action Input: {a['input']}"
                        key_phrase = "Action:"
                        last_index = output.rfind(key_phrase)
                        if last_index != -1:
                            output = output[last_index:].strip()

                    elif return_type == "give_up_and_restart":
                        print("⚠️ 放弃并重启：", a["input"].get("reason", "（无说明）"))
                        logger.info(f"⚠️ 放弃并重启： {a['input'].get('reason', '（无说明）')}")

                    elif return_type == "give_up":
                        print("❌ 放弃任务：", a["input"].get("reason", "（无说明）"))
                        logger.info(f"❌ 放弃任务： {a['input'].get('reason', '（无说明）')}")

                    else:
                        print("❓ 未知的 return_type:", return_type)
                        logger.info(f"❓ 未知的 return_type: {return_type}")
                    finish_hit=True
                    break

                result = env.step(a["action"], a["input"])
                print(f"🔧 执行工具 {a['action']} 成功，结果: {result}")
                logger.info(f"执行工具 {a['action']} 成功，结果: {result}")
                scratchpad += f"{output}\nObservation: {result}\n"

            except Exception as e:
                print("⚠️ 工具调用失败:", e)
                logger.info(f"工具调用失败: {e}")
                # finish_hit=True
                break
        if finish_hit:
            break

    print("turn:",turn)
    with open(args.score_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"step numbers:\t{turn+1}\n")

    if experience is not None and finish_hit:
        final_exp = add_experience(experience, scratchpad, query_data['query_id'], env, api_list, query)
    return output, solve_tokens, final_exp

if __name__ == "__main__":
    args = parse_args()
    with open(args.query_path, 'r', encoding='utf-8') as f:
        query_data_all = json.load(f)
    query_data = query_data_all[25]
    api_list = [{'category_name': 'eCommerce', 'tool_name': 'Amazon Pricing and Product Info', 'api_name': 'Main Endpoint'}, {'category_name': 'Financial', 'tool_name': 'Currency Converter_v2', 'api_name': 'Convert'}]
    tools_des, to_prune = get_tools_des(api_list)
    output = react_search(args, query_data, api_list, tools_des, 25, experience=None)
    print("===========================================")
    print(output)