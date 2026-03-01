import re
from tqdm import tqdm
import os
import json
import random
import ast
from termcolor import colored
from prompt import *
from models import gpt_exp
from config import model_name

current_dir = os.path.dirname(os.path.abspath(__file__))
tool_root_dir = os.path.join(current_dir, '..', 'tools')
tool_root_dir = os.path.abspath(tool_root_dir)  
with open('./data/tool_data.json', 'r', encoding='utf-8') as file:
    database = json.load(file)
with open('./data/api_details.json', 'r', encoding='utf-8') as file:
    api_details_dict = json.load(file)

def fix_truncated_response(truncated_response: str):
    truncated_response = truncated_response.replace('...','')
    
    try:
        response_dict = ast.literal_eval(truncated_response)
        return response_dict
    except (ValueError, SyntaxError):
        if truncated_response.endswith('}'):
            truncated_response = truncated_response[:-1] + '}]}'  
        elif truncated_response.endswith(']'):
            truncated_response += '}'  
        try:
            response_dict = ast.literal_eval(truncated_response)
            return response_dict
        except (ValueError, SyntaxError):
            return {"error": "无法修复格式"}
        
def standardize(string):
    if not isinstance(string, str):
        print('*'*100)
        print(string)
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string
  
def get_white_list(cache_path='./data/white_list_cache.json', force=False):
    """
    构建或加载工具白名单字典。

    参数：
    - tool_root_dir: 工具根目录，每个子目录为一个类别，包含多个工具 JSON 文件。
    - cache_path: 白名单缓存文件路径，默认保存在当前目录。
    - force: 是否强制重建缓存，默认为 False。

    返回：
    - white_list: dict 格式的白名单，key 为标准化后的 tool_name。
    """

    if not force and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    white_list = {}
    white_list_dir = os.path.join(tool_root_dir)

    for cate in tqdm(os.listdir(white_list_dir), desc="Building whitelist"):
        cate_path = os.path.join(white_list_dir, cate)
        if not os.path.isdir(cate_path):
            continue
        for file in os.listdir(cate_path):
            if not file.endswith(".json"):
                continue
            standard_tool_name = file.split(".")[0]
            file_path = os.path.join(cate_path, file)
            with open(file_path, encoding='utf-8') as reader:
                js_data = json.load(reader)
            try:
                origin_tool_name = js_data["tool_name"]
            except Exception as e:
                print('#' * 100)
                print('Error parsing file:', file_path)
                print('Exception:', e)
                print('Raw content:', js_data)
                continue

            white_list[standardize(origin_tool_name)] = {
                "description": js_data.get("tool_description", ""),
                "standard_tool_name": standard_tool_name
            }

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(white_list, f, ensure_ascii=False, indent=2)

    return white_list

def get_tools_des(api_list):
    if not api_list:
        return [], []
    origin_tool_names = [standardize(cont["tool_name"]) for cont in api_list]
    white_list = get_white_list()
    output = []
    to_prune = []
    for cand in origin_tool_names:
        if cand not in white_list.keys():
            to_prune.append[cand]
            continue
        output.append(white_list[cand])
    tools_des = [[cont["standard_tool_name"], cont["description"]] for cont in output]
    return tools_des, to_prune

def Finish():
    """Finish the conversation"""
    return 'finished'

def query_all_categories() -> list:
    """query all categories in the database"""
    return random.sample(list(database.keys()), len(database.keys()))

def get_tools_in_category(category_name: str=None) -> list:
    """query all tools in a specific category"""
    if category_name is None:
        return {'Error': 'Category name is required', 'response':''}
    if category_name not in database:
        return 'Illegal category name'
    return list(database[category_name].keys()) if category_name in database else None

def get_tools_descriptions(category_name: str, tool_list: str) -> dict:
    """query the details of a tool list"""
    if category_name not in api_details_dict:
        return {'Error': 'category name not found', 'response':''}
    if not isinstance(tool_list, list):
        return {'Error': 'tool_list must be a list', 'response':''}
    if isinstance(tool_list, str):
        tool_list = eval(tool_list)
    for tool_name in tool_list:
        if tool_name not in api_details_dict[category_name]:
            return f'tool name {tool_name} not found'
    return {tool_name: api_details_dict[category_name][tool_name]['description'] for tool_name in tool_list}

def query_all_tool_info(category: str, tools: list) -> dict:
    """
    查询指定类别下多个工具的 API 简要信息(只返回 name 和 description)及工具描述。
    """
    if not tools:
        return {'Error': 'Tool list is required', 'response': ''}
    if not isinstance(tools, list):
        return {'Error': 'Tools must be a list', 'response': ''}
    if category not in api_details_dict:
        return {'Error': f'Category {category} not found', 'response': ''}

    all_tools = api_details_dict[category]
    result = {}

    for tool in tools:
        if tool not in all_tools:
            return {'Error': f'Tool name {tool} not found in category {category}', 'response': ''}
        tool_info = all_tools[tool]
        api_list = tool_info.get("api_list", [])
        simplified_api_list = [
            {
                "name": api.get("name", ""),
                "description": api.get("description", "")
            }
            for api in api_list
        ]
        result[tool] = {
            "tool_description": tool_info.get("description", ""),
            "api_list": simplified_api_list
        }
    return result

def get_tool_description(category_name: str, tool_name: str) -> dict:
    """get the description of a specific tool"""
    if category_name not in api_details_dict:
        return 'category name not found'
    if tool_name not in api_details_dict[category_name]:
        return 'tool name not found'
    return api_details_dict[category_name][tool_name]

def get_api_details(category_name: str=None, tool_name: str=None, api_name: str=None) -> dict:
    """query the details of a specific api"""
    if category_name is None:
        return {'Error': 'Category name is required', 'response':''}
    if tool_name is None:
        return {'Error': 'Tool name is required', 'response':''}
    if api_name is None:
        return {'Error': 'API name is required', 'response':''}
    for category, tools in api_details_dict.items():
        if category != category_name:
            continue
        for tool, tool_data in tools.items():
            if tool != tool_name:
                continue
            for api in tool_data["api_list"]:
                if api["name"] == api_name:
                    return api
    return 'api not found'

def check_task_solvable_by_function_for_experience(query, functions, logger):
    messages = [{
        "role": "system",
        "content": CHECK_SOLVABLE_BY_FUNCTION_PROMPT_FOR_EXPERIENCE
    },
    {
        "role": "user",
        "content": f"Query: {query}. Available_tools: {functions}. Begin!"
    }]
    
    # logger.info(f"Check Solvable Prompt: {messages}")
    response = gpt_exp(messages=messages, temperature=0)
    content_str = response.choices[0].message.content

    try:
        content_json = json.loads(content_str)
        solvable = content_json.get("solvable", "")
        reason = content_json.get("reason", "")
        uncovered_subqueries = content_json.get("uncovered_subqueries", "") if solvable == "PartiallySolvable" else None
        
        print(colored(f"Solvable: {solvable}", 'light_cyan'))
        if reason:
            print("Reason:", reason)
        if uncovered_subqueries:
            print("Uncovered subqueries:", uncovered_subqueries)
        
        # logger.info(f"Solvable: {solvable}")
        # logger.info(f"Reason: {reason}")
        # if uncovered_subqueries:
        #     logger.info(f"Uncovered subqueries: {uncovered_subqueries}")
    
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        logger.info(f"JSON parsing error: {e}")
        solvable = "Unsolvable"
        reason = f"JSON parsing error: {e}"
        uncovered_subqueries = None
    
    return solvable, reason, uncovered_subqueries, response.usage.total_tokens

def evaluate_completion_score(query, system_response):
    prompt = EVALUATE_COMPLETION_SCORE_PROMPT.replace("{query}",query).replace("{system_response}",system_response)

    try:
        messages = [{"role": "user", "content": prompt}]
        response = gpt_exp(messages,model_name)
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
    except Exception as e:
        print(f"Error in evaluation: {e}")
        score = -1  # Use -1 to indicate evaluation failure

    return score

def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name

