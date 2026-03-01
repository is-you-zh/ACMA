import os
import json
import logging
from termcolor import colored
import time
import random
from copy import deepcopy
from collections import OrderedDict
import traceback
from retrieval import Retrieval_Agent
from exp_manager import ExperienceManagementAgent
from utils import *
from prompt import *
from models import *
from long_term_memory import *
from function_database import *
from config import *
from lats import lats_search
from react import react_search
from dfs import dfs_search
from classifier import select_strategy
from arguments import parse_args
args = parse_args()

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.log_path, exist_ok=True)

def init_logger(task_id: str, log_dir: str):
    logger = logging.getLogger(f'task_{task_id}')
    logger.setLevel(logging.INFO)
    log_filename = f"{log_dir}/task_{task_id}.log"

    if not logger.hasHandlers():
        handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        formatter = logging.Formatter(
            fmt='[%(asctime)s][%(message)s]',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def save_answer(args, query_data, action_param_json):
    with open(args.query_path, 'r', encoding='utf-8') as f:
        query_data_all = json.load(f)
    updated_dataset = []
    for item in query_data_all:
        if item.get("query_id") == query_data['query_id']:
            new_item = OrderedDict()
            for k, v in item.items():
                new_item[k] = v
                if k == "query_id":
                    new_item["final_answer"] = action_param_json["final_answer"]
            updated_dataset.append(new_item)
        else:
            updated_dataset.append(item)
    with open(args.query_path, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset, f, ensure_ascii=False, indent=2)

    print(f"Final Answer: {action_param_json.get('final_answer')}")
    logger.info(f"Final Answer: {action_param_json.get('final_answer')}")

def api_retriever(query, subtask):
    global global_api_list, retrieval_api_time, logger, retrieval_agent, retrieval_tokens
    print(colored('Retrieving API...', 'green'))
    logger.info('Retrieving API...')
    
    t_s = time.time()
    retrieval_agent = Retrieval_Agent(query=query, logger=logger, subtask=subtask)
    api_list, retrieval_tokens = retrieval_agent.execute()
    retrieval_api_time += (time.time() - t_s)
    unique_dict = {}
    global_api_list.extend(api_list)
    for item in global_api_list:
        if isinstance(item, str):
            try:
                item = json.loads(item)  
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for item: {item}. Error: {e}")
                continue  

        if isinstance(item, dict) and 'tool_name' in item and 'api_name' in item:
            unique_dict[f"{item['tool_name']}||{item['api_name']}"] = item
    else:
        print(f"Skipping invalid item: {item}")
    global_api_list = list(unique_dict.values())
    tools_des = prune_api()
    
    return tools_des
  
def prune_api():
    global global_api_list
    tools_des, to_prune = get_tools_des(global_api_list)
    if to_prune:
        print(colored(f'no tool description{to_prune}', 'red'))
        logger.info(f'no tool description{to_prune}')
        global_api_list = [cont for cont in global_api_list if cont["tool_name"] not in to_prune]

    print("Total API list size: "+str(len(global_api_list)))
    print("Retrieval API list: "+str(global_api_list))
    logger.info(f"Total API list size: {str(len(global_api_list))}\n"+
                'All API list detailed: ' + json.dumps(global_api_list, ensure_ascii=False, indent=2))   
    
    return tools_des
    
def run_lats(args, query_data, tools_des, i, experience):
    global global_api_list, final_exp
    selected_api_list = deepcopy(global_api_list)
    t_s = time.time()
    state, tokens_len, final_exp = lats_search(args, query_data, selected_api_list, i, tools_des, experience)
    
    with open(args.score_path, 'a', encoding='utf-8') as f:
        print(f'solve time:\t{time.time() - t_s:.4f}s', file=f)
    result = "Unsucessful question" if state is None else state['action']
    logger.info(f'result:{result}')
    action_name = result.split('{')[0] if '{' in result else result
    if action_name.startswith('functions.'):
        action_name = action_name[10:]
    action_param = result[result.find('{'):].strip() if '{' in result else ""
    action_param_json = json.loads(action_param)

    if action_name.lower() == 'finish':
        if isinstance(action_param_json, dict):
            if action_param_json.get('final_answer'):
                save_answer(args, query_data, action_param_json)
        else:
            logger.info(f"The answer is not a dict: {action_param_json}")

    return action_name, action_param_json, tokens_len, state

def run_dfs(args, query_data, tools_des, i, experience):
    global global_api_list, final_exp
    selected_api_list = deepcopy(global_api_list)
    t_s = time.time()
    solved, solve_data, tokens_len, output, final_exp = dfs_search(args, query_data, selected_api_list, i, tools_des, experience)
    with open(args.score_path, 'a', encoding='utf-8') as f:
        print(f'solve time:\t{time.time() - t_s:.4f}s', file=f)
    print("solved: ", solved, " result:", solve_data)
    action_param_json = solve_data
    if action_param_json['return_type'] == 'give_answer':
        save_answer(args, query_data, action_param_json)
    action_name = 'Finish'
    return action_name, action_param_json, tokens_len, output

def run_react(args, query_data, tools_des, i, experience):
    global global_api_list, final_exp
    selected_api_list = deepcopy(global_api_list)
    t_s = time.time()
    output, tokens_len, final_exp = react_search(args, query_data, selected_api_list, i, tools_des, experience)
    with open(args.score_path, 'a', encoding='utf-8') as f:
        print(f'solve time:\t{time.time() - t_s:.4f}s', file=f)
    action_match = re.search(r'Action:\s*(\w+)', output)
    action_input_match = re.search(r'Action Input:\s*({.*})', output, re.DOTALL)
    action_name = action_match.group(1).strip().lower() if action_match else None
    action_param = action_input_match.group(1).strip() if action_input_match else None
    action_param_json = {}
    try:
        action_param_json = json.loads(action_param)
    except Exception as e:
        logger.error(f"[run_react] Failed to parse action input JSON: {e}")
        logger.error(f"Raw action_param: {action_param}")
        return action_name, {} , tokens_len, output 
    action_param_str = json.dumps(action_param_json, ensure_ascii=False, indent=2)
    result = f"{action_name}({action_param_str})"
    logger.info(f'result:{result}')
    if action_name.startswith('functions.'):
        action_name = action_name[10:]
    if action_param_json.get('final_answer'):
        save_answer(args, query_data, action_param_json)

    return action_name, action_param_json, tokens_len, output
    
def reflection_and_resume_loop(new_exp, search_func, args):
    global global_api_list, failed_reason, query_data, query, pass_or_fail, retrieval_tokens
    t_start = time.time()
    tokens_len = 0
    output = None
    logger.info('return_type: restart')
    logger.info(f"reason: {failed_reason}")
    print(colored("Reflection and resume loop", 'green'))
    apis_to_remove = [api for api in global_api_list if api['api_name'] in str(failed_reason)]
    for api in apis_to_remove:
        print(colored(f'removing api: {api}', 'red'))
        global_api_list.remove(api)
    if apis_to_remove:
        print(f'APIs removed successfully. Current API number: {len(global_api_list)}. Max API number: {args.max_api_number}')
        logger.info(f'APIs removed successfully. Current API number: {len(global_api_list)}. Max API number: {args.max_api_number}')

    api_pool, tokens_len = retrieval_agent.resume_search(failed_reason,apis_to_remove)
    retrieval_tokens += tokens_len
    global_api_list.extend(api_pool)
    unique_dict = {
        f"{item['tool_name']}||{item['api_name']}": item 
        for item in global_api_list 
        if isinstance(item, dict) and 'tool_name' in item and 'api_name' in item
    }
    global_api_list = list(unique_dict.values())
    tools_des = prune_api()

    action_name, action_param_json, resume_solve_tokens, output = search_func(args, query_data, tools_des, i, new_exp)
    if action_name.lower() == 'finish':
        if action_param_json['return_type'] == 'give_answer':
            update_success_cnt(action_param_json, case=1)
            pass_or_fail = True
        else:
            update_success_cnt(action_param_json, case=2)
    else:
        update_success_cnt(action_param_json, case=3)
    print(f'reflection time:\t{time.time() - t_start:.4f}', file=open(args.score_path, 'a', encoding='utf-8')) 
    return resume_solve_tokens, output

def update_success_cnt(action_param_json, case=2):
    global query_data, unsolvable_task_cnt, success_task_cnt, valid_task_cnt, total_cnt
    if case==1:
        success_task_cnt += 1
        log_msg = (f'Solved, id: {query_data["query_id"]} '
                f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                f'success_task_cnt/valid_task_cnt: {success_task_cnt/valid_task_cnt}')
        with open(f'{args.output_dir}/success_cnt.txt', 'a', encoding='utf-8') as f:
            print(log_msg, file=f)      
    elif case==2:
        unsolvable_task_cnt += 1
        log_msg = (f'Unsolved, id: {query_data["query_id"]} reason: {action_param_json["reason"]} '
                f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                f'success_task_cnt/valid_task_cnt: {success_task_cnt/valid_task_cnt}')
        with open(f'{args.output_dir}/success_cnt.txt', 'a', encoding='utf-8') as f:
            print(log_msg, file=f)
        print(colored(f'Unsolved, reason: {action_param_json["reason"]}','red'))
        logger.info(f"reason: {action_param_json['reason']}")
    elif case==3:
        unsolvable_task_cnt += 1
        log_msg = (f'Unsolved, id: {query_data["query_id"]} reason: The number of available steps is insufficient. '
                f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                f'success_task_cnt/valid_task_cnt: {success_task_cnt/valid_task_cnt}')
        with open(f'{args.output_dir}/success_cnt.txt', 'a', encoding='utf-8') as f:
            print(log_msg, file=f)
        print(colored('Unsolved, the number of available steps is insufficient.','red'))
        logger.info('Unsolved, the number of available steps is insufficient.')
    elif case==4:
        unsolvable_task_cnt += 1
        log_msg = (f'Unsolved, id: {query_data["query_id"]} reason: The answer is not a dict. '
                f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                f'success_task_cnt/valid_task_cnt: {success_task_cnt/valid_task_cnt}')
        with open(f'{args.output_dir}/success_cnt.txt', 'a', encoding='utf-8') as f:
            print(log_msg, file=f)
        print(colored('Unsolved, the answer is not a dict.','red'))
        logger.info(f"The answer is not a dict: {action_param_json}")
    elif case==5:
        unsolvable_task_cnt += 1
        ratio = 0 if valid_task_cnt == 0 else success_task_cnt / valid_task_cnt
        log_msg = (f'Unsolvable checked by human, id: {query_data["query_id"]} '
                    f'unsolvable_cnt: {unsolvable_task_cnt} success_task_cnt: {success_task_cnt} '
                    f'valid_task_cnt: {valid_task_cnt} total_cnt: {total_cnt} '
                    f'success_task_cnt/valid_task_cnt: {ratio:.4f}')
        with open(os.path.join(output_dir, 'success_cnt.txt'), 'a', encoding='utf-8') as f:
            print(log_msg, file=f)
        print(colored('Unsolvable checked by human', 'yellow'))
        logger.warning('Unsolvable checked by human')
    else:
        print(colored(f'Updated case error: {case}', 'red'))
   

if __name__ == '__main__':
    unsolvable_task_cnt = 0
    success_task_cnt = 0
    valid_task_cnt = 0
    total_cnt = args.task_end_index - args.task_start_index
    unsolvable_list = json.load(open('./data/abnormal_queries.json', 'r', encoding='utf-8'))
    with open(args.query_path, 'r', encoding='utf-8') as f:
        query_data_all = json.load(f)

    # random.seed(42)
    # random.shuffle(query_data_all)
    print(f"total tasks: {len(query_data_all)}")
    new_task_list = []
    for task in query_data_all:
        output_file_path = os.path.join(args.log_path,f"task_{task.get('query_id')}.log")
        if not os.path.exists(output_file_path):
            new_task_list.append(task)
    query_data_all = new_task_list
    print(f"undo tasks: {len(query_data_all)}") 

    for i, query_data in enumerate(query_data_all):
        if query_data['query_id'] in unsolvable_list:
            update_success_cnt(query_data, case=5)
            continue

        global_api_list = []
        retrieval_tokens = 0
        solve_tokens = 0
        failed_reason = None
        experience = None
        tools_des = []
        value_output = None
        retrieval_api_time = 0
        experience_hit = False
        pass_or_fail = False
        valid_task_cnt += 1
        query = query_data['query']
        final_exp = None

        print(colored(f"task {i+1}/{len(query_data_all)}  real_task_id_{query_data.get('query_id')}","blue"))
        logger = init_logger(query_data['query_id'], args.log_path)
        logger.info(f'Query: {query}')
        print(colored(f'Query: {query}', 'green'))

        start_time = time.time()
        t_s = time.time()
        experience, new_exp = get_similar_experiences(query)
        print(time.time()-t_s)
        logger.info(f'Sence experience retrieval time: {time.time() - t_s:.2f}s')
        with open(args.score_path, 'a', encoding='utf-8') as f:
            print(f'\nquery id:\t{query_data["query_id"]}', file=f)
            print(f'retrieval sence experience time:\t{time.time() - t_s:.2f}s', file=f)

        if experience:
            try:
                exp_tools = experience.get("tool_details")
                exp_trajectory = experience.get("trajectory")
                for api in exp_tools:
                    if api not in global_api_list:
                        global_api_list.append(deepcopy(api))

            except Exception as e:
                print(colored(e,'red'))
                logger.info(f"{e}")      

            solvable, reason, sub_query, tokens_len = check_task_solvable_by_function_for_experience(query, global_api_list, logger)
            retrieval_tokens += tokens_len

            if solvable == 'FullySolvable':
                print(colored(f'Experience {experience.get("experience_id")} accepted.', 'green'))
                exp_str = json.dumps(experience, ensure_ascii=False, indent=2)
                print(exp_str)
                logger.info(exp_str)
                update_reuse_count_in_db(experience.get('experience_id'))
                experience_hit = True
                tools_des = prune_api()  

            elif solvable == 'PartiallySolvable':
                print(colored(f'Experience {experience.get("experience_id")} accepted.', 'green'))
                exp_str = json.dumps(experience, ensure_ascii=False, indent=2)
                print(exp_str)
                logger.info(exp_str)
                update_reuse_count_in_db(experience.get('experience_id'))
                experience_hit = True
                if(sub_query):
                    tools_des = api_retriever(query, sub_query)
                print(tools_des)
            
            else:
                print(colored('This experience is not applicable.', 'yellow'))           
                experience = None                   
                
        else:
            print(colored('This experience is not applicable.', 'yellow'))  
            experience = None   
            
        if not experience:
            global_api_list = []    
            tools_des = api_retriever(query, None)
            print(tools_des)

        algorithm = select_strategy(query, new_exp.subtasks, global_api_list, experience)
        print(colored(f"algorithm: {algorithm}",'blue'))
        logger.info(f"algorithm: {algorithm}")  

        if algorithm == 'lats':
            try:
                new_exp_copy = None if experience else new_exp
                action_name, action_param_json, tokens_len, value_output = run_lats(args, query_data, tools_des, i, new_exp_copy) 
                solve_tokens += tokens_len
                if action_param_json.get('return_type') == 'give_answer':
                    update_success_cnt(action_param_json, case=1)
                    pass_or_fail = True
                else:
                    if experience_hit:
                        global_api_list = []
                        tools_des = api_retriever(query, None)  
                        action_name, action_param_json, tokens_len, value_output = run_lats(args, query_data, tools_des, i, new_exp)
                        solve_tokens += tokens_len
                        if action_name.lower() == 'finish':
                            if action_param_json['return_type'] == 'give_answer':
                                update_success_cnt(action_param_json, case=1)
                                pass_or_fail = True
                            elif action_param_json['return_type'] == 'give_up':
                                update_success_cnt(action_param_json, case=2)
                            elif action_param_json['return_type'] == 'give_up_and_restart':
                                failed_reason = action_param_json['reason']
                                tokens_len, value_output = reflection_and_resume_loop(new_exp, run_lats, args)
                                solve_tokens += tokens_len
                            else:
                                update_success_cnt(action_param_json, case=2)
                        else:
                            update_success_cnt(action_param_json, case=3)
                    else:
                        if action_param_json['return_type'] == 'give_up':
                            update_success_cnt(action_param_json, case=2)
                        elif action_param_json['return_type'] == 'give_up_and_restart':
                            failed_reason = action_param_json['reason']
                            tokens_len, value_output = reflection_and_resume_loop(new_exp, run_lats, args)
                            solve_tokens += tokens_len
                        else:
                            update_success_cnt(action_param_json, case=2)
            
            except Exception as e:
                traceback.print_exc()
                logger.info(f"{e}")
                continue

        elif algorithm == 'dfs':
            try:
                new_exp_copy = None if experience else new_exp
                _, action_param_json,tokens_len,value_output = run_dfs(args, query_data, tools_des, i, new_exp_copy)
                solve_tokens+=tokens_len
                if action_param_json.get('return_type') == 'give_answer':
                    update_success_cnt(action_param_json, case=1)
                    pass_or_fail = True
                else:
                    if experience_hit:
                        global_api_list = []
                        tools_des = api_retriever(query, None)  
                        _, action_param_json,tokens_len,value_output = run_dfs(args, query_data, tools_des, i, experience=new_exp)
                        solve_tokens+=tokens_len
                        if action_param_json['return_type'] == 'give_answer':
                            update_success_cnt(action_param_json, case=1)
                            pass_or_fail = True
                        elif action_param_json['return_type'] == 'give_up':
                            update_success_cnt(action_param_json, case=2)
                        elif action_param_json['return_type'] == 'give_up_and_restart':
                            failed_reason = action_param_json['reason']
                            tokens_len, value_output = reflection_and_resume_loop(new_exp, run_dfs, args)
                            solve_tokens+=tokens_len
                        else:
                            update_success_cnt(action_param_json, case=2)
                    else:
                        if action_param_json['return_type'] == 'give_up':
                            update_success_cnt(action_param_json, case=2)
                        elif action_param_json['return_type'] == 'give_up_and_restart':
                            failed_reason = action_param_json['reason']
                            tokens_len, value_output = reflection_and_resume_loop(new_exp, run_dfs, args)
                            solve_tokens+=tokens_len
                        else:
                            update_success_cnt(action_param_json, case=2)

            except Exception as e:
                traceback.print_exc()
                logger.info(f"{e}")
                continue

        elif algorithm == 'react':
            try:
                new_exp_copy = None if experience else new_exp
                action_name, action_param_json, tokens_len, value_output = run_react(args, query_data, tools_des, i, new_exp_copy)
                solve_tokens += tokens_len
                if action_name.lower() == 'finish' and action_param_json.get('return_type') == 'give_answer':
                    update_success_cnt(action_param_json, case=1) 
                    pass_or_fail = True
                else:
                    if experience_hit:
                        global_api_list = []
                        tools_des = api_retriever(query, None)  
                        action_name, action_param_json, tokens_len, value_output = run_react(args, query_data, tools_des, i, experience=new_exp)
                        solve_tokens += tokens_len
                        if action_name == 'finish':
                            if isinstance(action_param_json, dict):
                                if action_param_json.get('return_type') == 'give_answer':
                                    update_success_cnt(action_param_json, case=1)
                                    pass_or_fail = True
                                elif action_param_json['return_type'] == 'give_up':
                                    update_success_cnt(action_param_json, case=2)
                                elif action_param_json['return_type'] == 'give_up_and_restart':
                                    failed_reason = action_param_json['reason']
                                    tokens_len, value_output = reflection_and_resume_loop(new_exp, run_react, args)
                                    solve_tokens += tokens_len
                            else:
                                update_success_cnt(action_param_json, case=4)
                        else:
                            update_success_cnt(action_param_json, case=3)
                    else:
                        if action_param_json['return_type'] == 'give_up':
                            update_success_cnt(action_param_json, case=2)
                        elif action_param_json['return_type'] == 'give_up_and_restart':
                            failed_reason = action_param_json['reason']
                            tokens_len, value_output = reflection_and_resume_loop(new_exp, run_react, args)
                            solve_tokens += tokens_len
                        else:
                            update_success_cnt(action_param_json, case=4)
            
            except Exception as e:
                traceback.print_exc()
                logger.info(f"{e}")
                continue

        else:
            raise Exception("Search algorithm option not valid")

        total_time = time.time() - start_time
        score = evaluate_completion_score(query, value_output)
        with open(f'{args.output_dir}/pass_rate.txt', 'a', encoding='utf-8') as f:
            print(f"{query_data['query_id']}: {True}", file=f)

        with open(args.score_path, 'a', encoding='utf-8') as f:
            print(f'retrieval api time:\t{retrieval_api_time:.4f}s', file=f)
            print(f'total time:\t{total_time:.4f}s', file=f)
            print(f'retrieval tokens:\t{retrieval_tokens}', file=f)
            print(f'solve tokens:\t{solve_tokens}', file=f)
            print(f'total tokens:\t{retrieval_tokens + solve_tokens}', file=f)
            print(f'score:\t{score}', file=f)

        print(colored("Next query.", 'green'))

        if pass_or_fail and final_exp:
            manager = ExperienceManagementAgent(final_exp,logger)
            manager.update_database()