import logging
import re
import numpy as np
import requests
import json
import time
from termcolor import colored
import toolenv
from models import gpt, gpt_no_tools
from tool_task import ToolTask,get_token_length
from long_term_memory import ExecutionStrategy, Metadata, ToolDetail
from config import *

class Node:
    def __init__(self, state, question, parent=None):
        self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False 
        self.em = 0  

    def uct(self):
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, thought={self.state['thought']}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }
    
def lats_search(args, query_data, api_list, i, tool_des, experience=None):
    global failed_trajectories, reflection_map, total_tokens
    failed_trajectories = []
    reflection_map = []
    total_tokens =0
    query = "Question:" + query_data['query']
    root = Node(state=None, question=query)
    all_nodes = []
    terminal_nodes = []
    final_exp = None

    task = ToolTask()
    env = toolenv.ToolEnv(query=query, api_list=api_list, tool_des=tool_des, process_id=i)

    global logger
    logger = logging.getLogger(f'task_{query_data["query_id"]}')

    seen_tools = set()
    for tool in tool_des:
        tool_name = tool[0]
        description = tool[1]
        if tool_name in seen_tools:
            continue  
        seen_tools.add(tool_name)
        # print(f"Tool Name: {tool_name}, Description: {description}")
        # logger.info(f"Tool Name: {tool_name}, Description: {description}")
    print(colored(f"Now executing {query}, with {len(api_list)} APIs", 'green'))

    for i in range(args.iterations):
        logger.info(f"Iteration {i + 1}...")
        node = select_node(root)
        print(colored(f"Execute the {i+1}-th round, selected node state:", 'green'))
        print(node.state)

        while node is None or (node.is_terminal and node.reward != 1):
            logger.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
            node = select_node(root)
        
        if node is None:
            logger.info("All paths lead to terminal nodes with reward 0. Ending search.")
            break

        if node.is_terminal and node.reward == 1:
            logger.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            if(experience is not None):
                final_exp = add_experience(experience, node, query_data['query_id'], env, api_list)
            return node.state, total_tokens, final_exp
        
        expand_node(node, args, task, tool_des, env)

        while node.is_terminal or not node.children:
            logger.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root)
            expand_node(node, args, task, tool_des, env)

        value = evaluate_node(node, args, task)
        reward, terminal_node = rollout(max(node.children, key=lambda child: child.value), args, task, max_depth=5, tool_des=tool_des, env=env)
        # 如果是finish函数，应该直接返回

        terminal_nodes.append(terminal_node)

        if terminal_node.reward == 1:
            logger.info("SUCCESSFUL TRAJECTORY FOUND DURING SIMULATION")
            if(experience is not None):
                final_exp = add_experience(experience, terminal_node, query_data['query_id'], env, api_list)
            return terminal_node.state, total_tokens, final_exp
        if terminal_node.is_terminal:
            logger.info("Refind Begin!")
            return terminal_node.state, total_tokens, final_exp
        
        # 如果是因为rollout超过了4层，reward就是0，不合理
        backpropagate(terminal_node, reward)
        all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

        terminal_nodes_with_reward_1 = [node for node in collect_all_nodes(root) if node.is_terminal and node.reward == 1]
        if terminal_nodes_with_reward_1:
            logger.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.value)
            if(experience is not None):
                final_exp = add_experience(experience, best_node, query_data['query_id'], env, api_list)
            return best_node.state, total_tokens, final_exp

        logger.info(f"Number of all_nodes after iteration {i + 1}: {len(all_nodes)}")

        for j, (node, value) in enumerate(all_nodes):
            logger.info(f"Node {j+1}: {str(node)}, value: {value}")

    all_nodes_list = collect_all_nodes(root)
    all_nodes_list.extend(terminal_nodes)
    best_child = max(all_nodes_list, key=lambda x: x.reward)
    failed_trajectories = []
    if best_child.reward == 1:
        if(experience is not None):
            final_exp = add_experience(experience, best_child, query_data['query_id'], env, api_list)
        logger.info("Successful trajectory found")
    else:
        logger.info("Unsuccessful trajectory found")
    if best_child is None:
        best_child = root
    return best_child.state, total_tokens

def select_node(node):
    while node and node.children:
        logger.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
     
        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]


        if len(terminal_children) == len(node.children):
            logger.info(f"All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:  
                node.parent.children.remove(node)
            node = node.parent  
            continue  
        
        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            logger.info(f"Found terminal node with reward 1 at depth {node.depth}.")
            return node_with_reward_1
        
        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

        while node.is_terminal and node.reward != 1:
            node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
        logger.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
    logger.info(f"final selected node: {str(node)}")
        
    return node  # This will return None if all paths from the root are exhausted

def expand_node(node, args, task, tool_des, env):
    if node.depth >= 8:
        print("Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, args.n_generate_sample, tool_des, env)
    node.children.extend(new_nodes)

def generate_new_states(node, args, task, n, tool_des, env):
    global failed_trajectories
    prompt = generate_prompt(node)
    # print(prompt)
    sampled_actions = get_samples(task, prompt, f"Thought {node.depth + 1}: ", n, stop="Observation", tool_des=tool_des, env=env, output_dir=args.output_dir)
    tried_actions = []
    unique_states = {}  
    i = 0
    for action in sampled_actions:
        i += 1
        new_state = node.state.copy()  
        depth = node.depth + 1

        pattern = re.compile(r'^(.*?)(?:^|\n)Action\s*[^:]*:(.*)', re.DOTALL)
        match = pattern.search(action)
        
        if match:
            # thought_line = match.group(1).strip()
            # action_line = match.group(2).strip()
            thought_line = match.group(1).strip()
            rest = match.group(2).strip()

            action_match = re.search(r'.*?\}', rest, re.DOTALL)
            if action_match:
                action_line = action_match.group(0).strip()
            else:
                action_line = rest  
        else:
            thought_line = action.strip()
            action_line = ""
            
        # print("thought_line:",thought_line,"\n\n","action_line:",action_line)
        
        if action_line in unique_states:
            logger.info(f"SAMPLED ACTION {i}:Action already exists")
            continue  
        else:
            logger.info(f"SAMPLED ACTION {i}:{action}")     
            print("thought_line:",thought_line)       

        tried_actions.append(action_line)
        if action_line:
            action_name = action_line.split('{')[0] if '{' in action_line else action_line
            action_param = action_line[action_line.find('{'):].strip() if '{' in action_line else ""
            action_param_json = json.dumps(action_param)
            # print(action_name, action_param)
            if action_name.startswith('functions.'):
                action_name = action_name[10:]
            obs, status, r, done, info = step(env, action_name, action_param)
            # print("generate new node")

            new_state['thought'] = thought_line
            new_state['action'] = action_line
            if isinstance(obs, str) and len(obs) > 1500:
                obs = obs[:1500] + "..."
            new_state['observation'] = obs

            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done
            new_node.reward = r
            new_node.depth = node.depth + 1
            unique_states[action_line] = new_node  
            print(f"🔧 执行工具 {action_name} 成功，结果: {obs}")
            logger.info(f"OBSERVATION: {obs} \nFeedback: {info}")

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory(new_node)
                failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_name.lower()}[{action_param}]", 'reason': info["answer"]})
            if action_name.lower()=='finish':
                with open(args.score_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"step numbers:\t{info.get('steps')}\n")
                    return list(unique_states.values()) 
    return list(unique_states.values())  

def get_samples(task, node_info, sentence, n_generate_sample, stop, tool_des, env, output_dir):
    global failed_trajectories, total_tokens, reflection_map
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    if len(unique_trajectories) > len(reflection_map) and len(unique_trajectories) < 4:
        print("generating reflections")
        reflection_map = task.generate_self_reflection(unique_trajectories, node_info, tool_des, env.functions)
    try:
        prompt = task.cot_prompt_wrap_with_tool(node_info, sentence, reflection_map, tool_des)
    except Exception as e:
        print(f"❌ Failed to generate prompt: {e}")
        prompt = ""
    logger.info(f"PROMPT: {prompt}")
    tools = env.converted_list
    # logger.info(f"tools: {tools}")
    tools = [tool["function"] for tool in tools]
    samples,tokens = gpt(prompt, model=model_name, n=n_generate_sample, stop=stop, tools=tools)
    total_tokens += tokens
    return [sentence + str(_) for _ in samples]

def get_unique_trajectories(failed_trajectories, num=5):
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get('final_answer')
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj['trajectory']))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories

def node_trajectory_to_text(node_string):
    lines = node_string.split('\n')
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            thought = line.split(", thought=")[1].split(", action=")[0].strip()
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except IndexError:
            continue
        
        if depth != 0:
            if thought:
                formatted_lines.append(f"Thought {depth}: {thought}")
            if action:
                formatted_lines.append(f"Action {depth}: {action}")
            if observation:
                formatted_lines.append(f"Observation {depth}: {observation}")
    
    return '\n'.join(formatted_lines)

def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['thought']:
            new_segment.append(f"Thought {node.depth}: {node.state['thought']}")
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n'.join(reversed(trajectory))

def step(env, action_name, action_input):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action_name, action_input)
        except requests.exceptions.Timeout:
            attempts += 1

def get_action_name_trajectory(node):
    trajectory = []
    func_names = []
    question = node.question

    while node:
        action_str = node.state.get('action', '').strip()
        if action_str:
            trajectory.append(f"Action {node.depth}: {action_str}")
            fun_name = action_str
            if action_str.startswith('functions.'):
                fun_name = action_str[len('functions.'):]
            brace_index = fun_name.find('{')
            if brace_index != -1:
                fun_name = fun_name[:brace_index].strip()
            func_names.append(fun_name)
        node = node.parent
    trajectory.append(question)

    return list(reversed(trajectory)), func_names

def collect_trajectory(node):
    trajectory = []
    while node:
        trajectory.append(str(node))
        node = node.parent
    return '\n'.join(reversed(trajectory))

def evaluate_node(node, args, task):
    target_children = [child for child in node.children if not child.is_terminal]
    children_prompt = []
    logger.info(f"Length of node.children: {len(node.children)}")
    for i, child in enumerate(target_children, 1):
        children_prompt.append(f'the {i} child node:')
        if child.state['thought']:
            children_prompt.append(f"Thought {child.depth}: {child.state['thought']}")
        if child.state['action']:
            children_prompt.append(f"Action {child.depth}: {child.state['action']}")
        if child.state['observation'] and child.depth != 0:
            children_prompt.append(f"Observation {child.depth}: {child.state['observation']}")
        children_prompt.append("")
    if children_prompt:
        if get_token_length(children_prompt) < 5000:
            trajectory = generate_prompt(node)
            prompt = trajectory + '\n' +str(children_prompt)
            values = get_values_together(task, node.question, prompt,args.n_evaluate_sample, output_dir=args.output_dir)
            return sum(values) / len(node.children) if values else 0
        else:
            logger.info("tokens number exceed the max number")
            child_prompts = [generate_prompt(child) for child in target_children]
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample, output_dir=args.output_dir)
            for i, child in enumerate(node.children):
                child.value = values[i] 
            return sum(values) / len(node.children) if values else 0
    else:
        return 0

def get_values_together(task, x, y, n_evaluate_sample, output_dir, cache_value=True):
    global total_tokens
    value_prompt = task.value_prompt_wrap_together(x, y)
    # logger.info(f"VALUE PROMPT: {value_prompt}")
    value_outputs, tokens = gpt_no_tools(value_prompt, model=model_name, n=n_evaluate_sample, stop=None)
    total_tokens += tokens
    print('values:', value_outputs)
    extracted_list = eval(value_outputs[0])
    values = task.value_outputs_unwrap_together(extracted_list)
    logger.info(f"VALUES: {values}")
    return values

def get_values(task, x, ys, n_evaluate_sample, output_dir, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  
        if y in local_value_cache:  
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, output_dir, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_value(task, x, y, n_evaluate_sample, output_dir, cache_value=True):
    global reflection_map, total_tokens
    global failed_trajectories
    
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    value_prompt = task.value_prompt_wrap(x, y, unique_trajectories, reflection_map)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    # logger.info(f"VALUE PROMPT: {value_prompt}")
    value_outputs, tokens = gpt_no_tools(value_prompt, model=model_name, n=n_evaluate_sample, stop=None)
    
    total_tokens += tokens
    value = task.value_outputs_unwrap(value_outputs)
    logger.info(f"VALUES: {value}")
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def rollout(node, args, task, max_depth, tool_des, env):
    depth = node.depth
    n = 5
    rewards = [0]
    while not node.is_terminal and depth < max_depth:
        logger.info(f"ROLLING OUT DEPTH: {depth}")
        new_states = []
        values = []
        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, n, tool_des, env)

        for state in new_states:
            if state.is_terminal:
                logger.info(f"Return terminal node: {state}")
                return state.reward, state
                
        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        while len(values) == 0:
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample, output_dir=args.output_dir)
            logger.info(f"values: {values}")
        max_value_index = values.index(max(values))
        rewards.append(max(values))
        logger.info(f'rewards append {max(values)}')
        node = new_states[max_value_index] 
        depth += 1
    
    logger.info("ROLLOUT FINISHED")
    logger.info(f"sum(rewards) / len(rewards): {sum(rewards) / len(rewards)}")
    return sum(rewards) / len(rewards), node

def backpropagate(node, value):
    while node:
        node.visits += 1
        if node.is_terminal:
            if node.reward == 0:
                node.value = (node.value * (node.visits - 1) + (-1)) / node.visits
                logger.info(f"Backpropagating with reward 0 at depth {node.depth}. New value: {node.value}.")
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits
                logger.info(f"Backpropagating with reward 1 at depth {node.depth}. New value: {node.value}.")
        else:
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logger.info(f"Backpropagating at depth {node.depth}. New value: {node.value}.")

        node = node.parent

def collect_all_nodes(node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(collect_all_nodes(child))

        return nodes

def add_experience(experience, node, query_id, env, api_list):
    tool_order, used_api = get_action_name_trajectory(node)
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
    # experience.tool_details = [ToolDetail(**d) for d in selected_api_list]
    experience.tool_details = [
        ToolDetail(**{k: v for k, v in d.items() if k != 'description'}) 
        for d in selected_api_list
    ]
    print(experience)

    # add_experience_to_db(experience)
    logger.info(f"[INFO] The experience was added successfully: {experience}")
    return experience