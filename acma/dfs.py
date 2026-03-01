import json
import time
from long_term_memory import ExecutionStrategy, Metadata, ToolDetail
import re
from config import *
from utils import *
from prompt import *
from models import call_gpt
import logging
import os
from termcolor import colored
from copy import deepcopy
from tenacity import retry, wait_random_exponential, stop_after_attempt
from toolenv import ToolEnv
from arguments import parse_args
args = parse_args()
output_dir = args.output_dir
score_path = args.score_path

global FINAL_TRAJECTORY, FINAL_FUNC_NAMES, logger, dfs_args
logger = None

class tree_node:
    def __init__(self):
        self.is_terminal = False
        self.pruned = False
        self.finished = False
        self.node_type = None
        self.description = ""
        self.observation = ""
        self.observation_code = None
        self.children = []
        self.father = None
        self.io_state = None
        self.expand_num = 0 # The number of visits to the node, 0 means it has not been visited
        self.Elo = 1000.0
        self.messages = []

    def get_max_depth(self):
        '''
        maximum depth of subtrees including self
        '''
        max_depth = 0
        for child in self.children:
            max_depth = max(max_depth,child.get_max_depth())
        return max_depth + 1

    def get_depth(self):
        if self.father == None:
            return 0
        return self.father.get_depth() + 1

    def get_size(self):
        '''
        subtree, including itself
        '''
        size = 1
        for child in self.children:
            size += child.get_size()
        return size
    
    def prune(self):
        '''
        pruning off the subtree
        '''
        self.pruned = True
        for child in self.children:
            child.prune()

    def print(self):
        color_converter = {"Thought":"red", "Action": "blue", "Action Input": "cyan","Final Answer": "green","Reflection":"blue"}
        print(colored(f"{self.node_type}: {self.description}",color = color_converter[self.node_type]))
        logger.info(f"{self.node_type}: {self.description}")
        if self.observation != "":
            if len(self.observation) < 1500:
                print(colored(f"Observation: {self.observation}",color="yellow"))
                logger.info(f"Observation: {self.observation}")
            else:
                print(colored(f"Observation: {self.observation[:1536]}......(len={len(self.observation)})",color="yellow"))
                logger.info(f"Observation: {self.observation[:1536]}......(len={len(self.observation)})")

    @classmethod
    def find_ancestor_intersection(cls, node1, node2):
        '''
        find the first common ancestor
        '''
        if node1 == None or node2 == None:
            return None
        if node1 == node2:
            return node1
        length1 = node1.get_depth()
        length2 = node2.get_depth()
        if length1 > length2:
            return tree_node.find_ancestor_intersection(node1.father,node2)
        else:
            return tree_node.find_ancestor_intersection(node1, node2.father)

    def make_finish(self,inter_val=1):
        '''
        Recursively marked as finish, until the above inter_val nodes of action_input type (including yourself)
        '''
        self.finished = True
        if self.node_type == "Action Input":
            inter_val -= 1
        if self.father != None and inter_val >= 0:
            self.father.make_finish(inter_val)

    def get_train_messages_from_this_node(self):
        '''
        Returns chained results, starting from this node up to the root node
        '''
        def sift_first_invalid_message(messages):
            use_messages = []
            flag = True
            for message_id in range(len(messages))[::-1]:
                if not ("valid" in messages[message_id].keys() and messages[message_id]["valid"] == False):
                    use_messages = [messages[message_id]] + use_messages
                elif flag:
                    flag = False
                    use_messages = [messages[message_id]] + use_messages
            return use_messages

        now_node = self
        result = []
        while now_node.father != None:
            if now_node.node_type == "Action Input":
                use_messages = deepcopy(now_node.messages)
                while use_messages[-1]["role"] != "assistant":
                    use_messages = use_messages[:-1]
                use_messages = sift_first_invalid_message(use_messages)
                result = [use_messages] + result
            elif now_node.node_type == "Thought":
                use_messages = deepcopy(now_node.messages)
                while use_messages[-1]["role"] == "user":
                    use_messages = use_messages[:-1]
                use_messages = sift_first_invalid_message(use_messages)
                if use_messages[-1]["role"] == "assistant":
                    result = [use_messages] + result
            now_node = now_node.father
        return result

    def get_chain_result_from_this_node(self,use_messages=False):
        '''
        Returns chained results, starting from this node up to the root node
        '''
        now_node = self
        result = []
        while now_node.father != None:
            result = [now_node.to_json(use_messages=use_messages)] + result
            now_node = now_node.father
        return result

    def get_former_trice_from_this_node(self,valid_types=["Thought","Action","Action Input","Observation"],end_node = None):
        '''
        Return path description from end_node -> self
        Does not contain end_node, never contains root node
        '''
        node = self
        output_str_list = []

        while node != end_node and node.father != None:
            now_node_des_list = []
            if node.node_type in valid_types:
                now_node_des_list.append(f"{node.node_type}: {node.description}\n")
            if node.observation != "" and "Observation" in valid_types:
                tuncated = node.observation
                if len(node.observation) > 1024:
                    tuncated = node.observation[:1024] + f"...(len={len(node.observation)})"
                now_node_des_list.append(f"Observation: {tuncated}\n")
            output_str_list = now_node_des_list + output_str_list
            node = node.father
        
        now_str = ""
        for k, cont in enumerate(output_str_list):
            now_str += f"step_{k+1}: {cont}\n"

        if now_str == "":
            now_str = "None"
        return now_str

    def to_json(self, use_messages=False):
        
        json_obj = {}
        json_obj["is_terminal"] = False
        json_obj["pruned"] = self.pruned
        json_obj["finished"] = self.finished

        json_obj["depth"] = self.get_depth()
        json_obj["node_type"] = self.node_type
        json_obj["description"] = self.description
        json_obj["Elo"] = self.Elo
        if self.observation != "":
            json_obj["observation"] = self.observation
        if self.observation_code != None:
            json_obj["observation_code"] = self.observation_code
        json_obj["child_count"] = len(self.children)
        json_obj["expand_num"] = self.expand_num

        if self.io_state != None and self.node_type == "Action Input":
            json_obj["io_state"] = self.io_state.to_json()

            
        if use_messages:
            json_obj["messages"] = []
            for message in self.messages:
                if not ("valid" in message.keys() and message["valid"] == False):
                    json_obj["messages"].append(message["role"])
                else:
                    json_obj["messages"].append(message["role"] + "_invalid")

        return json_obj

    def to_json_recursive(self,use_messages=False):
        js_obj = self.to_json(use_messages=use_messages)
        js_obj["children"] = []
        for child in self.children:
            js_obj["children"].append(child.to_json_recursive())
        return js_obj
    
    
class my_tree:
    def __init__(self):
        self.root = tree_node()
        self.now_deal_node = self.root

    def to_json_recursive(self,use_messages=False):
        tree_structure =  self.root.to_json_recursive(use_messages=use_messages)
        js_obj = {
            "size": self.root.get_size(),
            "max_length":self.root.get_max_depth(),
            "tree": tree_structure,
        }
        return js_obj
    

class DFS_tree_search():
    def __init__(self, llm, io_func, query=None):
        """Depth-first search. 
        with_filter=True: Every time a child node is generated, choose the best multiple iterations to go.
        with_filter=False: Do as Preorder traversal.
        """
        self.io_func = io_func
        self.llm = llm
        self.restart()
        self.query = query

    def restart(self):
        self.status = 0
        self.terminal_node = []
        self.give_up_node = []
        self.now_expand_num = 0
        self.query_count = 0
        self.total_tokens = 0

    def to_json(self, answer=False, process=True):

        if process:
            json_obj = {
                "win": self.status == 1,
                "tree": self.tree.to_json_recursive(),
                "forward_args": self.forward_args,
                "compare_candidates": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:  
                    json_obj["compare_candidates"].append(
                        node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "final_answer": "",
                "finish_type": "give_answer",
                "function": self.io_func.functions,
                "chain": [],
            }
            for node in self.terminal_node:
                if node.pruned == False:
                    if 'give_up' in node.description.lower():
                        json_obj["answer_generation"]["finish_type"] = "give_up"
                    else:
                        json_obj["answer_generation"]["finish_type"] = "give_answer"
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
            # do not have final answer, look for give_up
            if json_obj["answer_generation"]["valid_data"] == False:
                if len(self.give_up_node) > 0:
                    random_pos = random.randint(0, len(self.give_up_node) - 1)
                    choose_give_up_node = self.give_up_node[random_pos]
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_up"
                    json_obj["answer_generation"]["final_answer"] = choose_give_up_node.description
                    json_obj["answer_generation"]["train_messages"] = choose_give_up_node.get_train_messages_from_this_node()
        return json_obj

    def start(self, single_chain_max_step, tree_beam_size, max_query_count, answer=1, with_filter=True, messages=None):
        """ single_chain_max_step: The maximum depth of the tree
            tree_beam_size: How many children nodes for one node are generated per layer
            answer = n means the Algo exits when find n "give_answer" nodes
            max_query_count: the Algo exits when OpenAI-query exists this value
            with_filter: This is the difference between normal DFS(with_filter=True) and DFSDT(with_filter=False). 
        """
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args=None
        self.tree = my_tree()
        self.tree.root.node_type = "Action Input"
        self.tree.root.io_state = deepcopy(self.io_func)
        system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ADAPTED.replace("{task_description}", self.io_func.task_description)
        user = FORMAT_INSTRUCTIONS_USER_FUNCTION.replace("{input_description}", self.io_func.input_description)
        if messages is None:
            self.tree.root.messages.append({"role": "system", "content": system})
            self.tree.root.messages.append({"role": "user", "content": user})
            logger.info({"role": "system", "content": system})
            logger.info({"role": "user", "content": user})
        else:
            messages[0] = {"role": "system", "content": system}
            messages.pop()
            function_names = []
            for function in self.io_func.functions:
                function_names.append(function["name"])
            for i, message in reversed(list(enumerate(messages))):
                if message["role"] == "assistant":
                    if message["name"] not in function_names:
                        messages.pop(i)
                        messages.pop(i-1)
                if message["role"] == "user":
                    if 'maximum query count' in message["content"]:
                        messages.pop(i)

            self.tree.root.messages = messages

        result = self.DFS(self.tree.root, single_chain_max_step, tree_beam_size, max_query_count, answer, with_filter)
        return result,self.total_tokens

    def DFS(self, now_node, single_chain_max_step, tree_beam_size, max_query_count, answer, with_filter=True):
        """Returns the number of grids to go back. When a child node of a node generates a final answer or give up, it should go back a few more grids
        In a sense, the larger this value is, the more diverse it is, and it is GreedySearch@n when it is enlarged to infinity.
        """
        # this two value declares the rate to go back, Algo degrades to CoT when the value=Inf
        final_answer_back_length = 2
        prune_back_length = 2

        now_node.expand_num = self.now_expand_num
        self.now_expand_num += 1
        if now_node.get_depth() >= single_chain_max_step or now_node.pruned or now_node.is_terminal:
            if now_node.is_terminal:  
                self.status = 1
                self.terminal_node.append(now_node)
                get_clean_trajectory_with_action_args(now_node, self.query)
                return final_answer_back_length
            else:
                now_node.pruned = True
                if now_node.observation_code == 4:
                    self.give_up_node.append(now_node)
                    return prune_back_length
                else:
                    return 1

        next_tree_split_nodes = []
        for i in range(tree_beam_size):
            temp_now_node = now_node

            """If a node have children now, We will prompt the model to generate different nodes than all the existing nodes"""
            delete_former_diversity_message = False
            diversity_message = None
            if len(temp_now_node.children) > 0:

                former_candidates_des = ""
                js_list = []
                for k, child in enumerate(temp_now_node.children):
                    temp_node = child
                    while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                        temp_node = temp_node.children[0]
                    if temp_node.node_type == "Action Input":
                        obj_dict = {
                            "name": temp_node.father.description,
                            "arguments": temp_node.description,
                            "function_output": temp_node.observation
                        }
                        js_list.append(obj_dict)

                if len(js_list) > 0:
                    former_candidates_des = former_candidates_des + \
                        f"{json.dumps(js_list,indent=2)}\n"
                    if temp_now_node.observation != "":
                        former_candidates_des = former_candidates_des + \
                            f"again, your former observation: {temp_now_node.observation}\n"
                    diverse_prompt = DIVERSITY_PROMPT
                    diverse_prompt = diverse_prompt.replace(
                        "{previous_candidate}", former_candidates_des)
                    diversity_message = {
                        "role": "user", "content": diverse_prompt}
                    temp_now_node.messages.append(diversity_message)

                    delete_former_diversity_message = True
            self.llm.change_messages(temp_now_node.messages)
            new_message, error_code, total_tokens = self.llm.parse(functions=self.io_func.functions)
            self.query_count += 1
            self.total_tokens += total_tokens

            # We need to exclude the diversity_message, because it will influence child nodes
            if delete_former_diversity_message:
                temp_now_node.messages[-1]["valid"] = False

            # parse nodes from OpenAI-message like CoT method
            assert new_message["role"] == "assistant"
            if "content" in new_message.keys() and new_message["content"]:
                temp_node = tree_node()
                temp_node.node_type = "Thought"
                temp_node.description = new_message["content"]
                child_io_state = deepcopy(temp_now_node.io_state)
                child_io_state.retriever=None

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)
                temp_node.print()
                temp_now_node = temp_node

                if error_code != 0:
                    temp_now_node.observation_code = error_code
                    temp_now_node.pruned = True

            if "function_call" in new_message.keys() and new_message["function_call"] is not None:
                function_name = new_message["function_call"]["name"]
                temp_node = tree_node()
                temp_node.node_type = "Action"
                temp_node.description = function_name
                child_io_state = deepcopy(temp_now_node.io_state)
                child_io_state.retriever=None

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)

                temp_node.print()
                temp_now_node = temp_node

                function_input = new_message["function_call"]["arguments"]
                temp_node = tree_node()
                temp_node.node_type = "Action Input"
                temp_node.description = function_input
                child_io_state = deepcopy(temp_now_node.io_state)
                child_io_state.retriever=None
                observation, status, _, _, _ = child_io_state.step(
                    action_name=temp_now_node.description, action_input=function_input)
                if status == 1:
                    print(observation)
                    logger.info(f"Observation: {observation}")
                temp_node.observation = observation
                temp_node.observation_code = status

                temp_node.io_state = child_io_state
                temp_node.is_terminal = child_io_state.check_success() != 0
                temp_node.messages = deepcopy(temp_now_node.messages)
                temp_node.father = temp_now_node
                temp_now_node.children.append(temp_node)
                temp_node.print()
                temp_now_node = temp_node
                if status != 0:
                    if status == 4:
                        temp_now_node.pruned = True
                    elif status == 1:
                        assert "function_call" in new_message.keys()
                        print(new_message["function_call"]["name"], file=open(f'{output_dir}/hallucination.txt','a'))
                        logger.info(f"Hallucination: {new_message['function_call']['name']}")
                        new_message["function_call"]["name"] = "invalid_hallucination_function_name"
                    elif status == 3: 
                        temp_now_node.is_terminal = True
                        temp_now_node.make_finish(final_answer_back_length)

            temp_now_node.messages.append(new_message)
            if temp_now_node.node_type == "Action Input":
                temp_now_node.messages.append({
                    "role": "assistant",
                    "name": new_message["function_call"]["name"] if new_message["function_call"] else '',
                    "content": temp_now_node.observation,
                })
            if self.query_count >= max_query_count:
                temp_now_node.messages.append({
                    "role": "user",
                    "content": "you have reached the maximum query count, please call the finish function to give the answer or give up without restart.",
                })
            return_value = None
            if not with_filter: 
                result = self.DFS(temp_now_node, single_chain_max_step,
                                  tree_beam_size, max_query_count, answer, with_filter)
                if len(self.terminal_node) >= answer:
                    return_value = 10000
                elif result > 1:
                    return_value = result-1

            else:
                next_tree_split_nodes.append(temp_now_node)
            if return_value is not None:
                return return_value

        # Sort the generated next_tree_split_nodes nodes when normal DFS
        if len(next_tree_split_nodes) > 1:
            # When using normal DFS, if we have many child nodes, we will refer to LLM to compare and choose the best one to expand first
            # remember, this operator will cost extra OpenAI calls.
            LLM_rank_args = {
                "functions": self.io_func.functions,
                "task_description": self.io_func.task_description,
                "rank_func": rank2_subfix,
            }
            scores, rank_query_count, total_tokens = sum_based_rankn(
                self.llm, LLM_rank_args=LLM_rank_args, candidates=next_tree_split_nodes)
            self.query_count += rank_query_count
            self.total_tokens += total_tokens
            for score, node in zip(scores, next_tree_split_nodes):
                node.prior_score = score
            zip_value = list(
                zip(next_tree_split_nodes, range(len(next_tree_split_nodes))))
            zip_value.sort(
                key=lambda x: x[0].prior_score, reverse=True)  # 先做score高的
            next_tree_split_nodes, filtered_order = zip(*zip_value)

        '''
        Choose one to expand
        '''
        for i in range(len(next_tree_split_nodes)):
            result = self.DFS(
                next_tree_split_nodes[i], single_chain_max_step, tree_beam_size, max_query_count, answer)
            if len(self.terminal_node) >= answer:
                return 10000
            elif result > 1:
                now_node.make_finish(2)
                return result - 1

        return 1


class pipeline_runner:
    def __init__(self):
        pass

    def method_converter(self, method, env, single_chain_max_step=12, max_query_count=60, messages=None, query=None):
        llm_forward = GPT4Function()
        pattern = r".+_w(\d+)"
        re_result = re.match(pattern,method)
        assert re_result != None
        width = int(re_result.group(1))
        with_filter = True
        if "woFilter" in method:
            with_filter = False
        chain = DFS_tree_search(llm=llm_forward, io_func=env, query=query)
        result, tokens_len = chain.start(
                            single_chain_max_step=single_chain_max_step,
                            tree_beam_size = width,
                            max_query_count = max_query_count,
                            answer=1,
                            with_filter=with_filter,
                            messages=messages)

        return chain, result, tokens_len

    def run(self, method, data_dict, output_dir_path, tool_des, messages, env):
        splits = output_dir_path.split("/")
        os.makedirs("/".join(splits),exist_ok=True)
        query = data_dict["query"]
        print(colored(f"Now playing: {query}, with {len(env.functions)} APIs", "green"))
        chain,result,tokens_len = self.method_converter(
            method=method,
            env=env,
            single_chain_max_step=12,
            max_query_count=10,
            messages=messages,
            query=query
        )
        if output_dir_path is not None:
            print('#'*100)
            data = chain.to_json(answer=True,process=True)
            data["answer_generation"]["query"] = query
            data["api2origin"] = env.api2origin
            success = data["answer_generation"]["valid_data"] and "give_answer" in data.get("answer_generation", {}).get("final_answer", "")
            print(colored(f"valid={success}", "green"))
            logger.info(f"valid={success}")
            output = data["answer_generation"]["final_answer"]
            with open(score_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"step numbers:\t{data.get('tree').get('size')}\n")              
            result = data.get("answer_generation", {}).get("final_answer", "")
        return result,tokens_len,output


class GPT4Function:
    def __init__(self):
        self.conversation_history = []
        self.TRY_TIME = 6

    def change_messages(self,messages):
        self.conversation_history = messages

    def parse(self,functions,**args):
        for conversation in self.conversation_history:
            if 'content' not in conversation:
                conversation['content'] = ''
                with open(f"{output_dir}/error.txt", 'a', encoding='utf-8') as f:
                    f.write(f"Error Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Found missing 'content' in conversation history.\n")
                    f.write(f"Full conversation_history context:\n{self.conversation_history}\n")
                # print(self.conversation_history, file=open('tmp.txt','a'))
        conversation_history = self.conversation_history
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)
            json_data = chat_completion_request(conversation_history, functions=functions)
            try:
                total_tokens = json_data.usage.total_tokens
                print(f"total tokens: {total_tokens}")
                logger.info(f"total tokens: {total_tokens}")
                message_obj = json_data.choices[0].message
                func_call_data = message_obj.function_call
                if func_call_data and "." in func_call_data.name:
                    func_call_dict = func_call_data.model_dump()
                    func_call_dict["name"] = func_call_dict["name"].split(".")[-1]
                else:
                    func_call_dict = func_call_data.model_dump() if func_call_data else None

                result_message = {
                    "role": message_obj.role,
                    "function_call": func_call_dict,
                    "content": message_obj.content if message_obj.content else ''
                }

                return result_message, 0, total_tokens

            except BaseException as e:
                print(f"Parsing Exception: {repr(e)}. Try again.")
                logger.info(f"Parsing Exception: {repr(e)}. Try again. ")
                if json_data is not None:
                    print(f"OpenAI return: {json_data}")
                    logger.info(f"OpenAI return: {json_data}")
            

        return {"role": "assistant", "content": str(json_data)}, -1, 0

def rank2_subfix(llm_interface,LLM_rank_args, cand1,cand2):
    '''
    Assumed that the two candidates have a long common prefix
    '''
    anscestor_interesction = tree_node.find_ancestor_intersection(cand1,cand2)
    assert anscestor_interesction != None
    intersect_trice = anscestor_interesction.get_former_trice_from_this_node(end_node=None)
    trice_1 = cand1.get_former_trice_from_this_node(end_node=anscestor_interesction)
    trice_2 = cand2.get_former_trice_from_this_node(end_node=anscestor_interesction)

    system_message = LLM_PAIRWISE_RANK_SUBFIX_SYSTEM_PROMPT
    system_message = system_message.replace("{task_description}", LLM_rank_args["task_description"])
    system_message = system_message.replace("{intersect_trice}", intersect_trice)
    system_message = system_message.replace("{candidate_A}",trice_1)
    system_message = system_message.replace("{candidate_B}",trice_2)
    llm_interface.change_messages([{"role":"system","content":system_message},
                                   {"role":"user","content":LLM_PAIRWISE_RANK_USER_PROMPT},
                                   ])
    output,error_code, total_tokens = llm_interface.parse(functions=LLM_rank_args["functions"])
    if output["content"].strip().lower()[-1] == "a":
        return 1, 1, total_tokens
    else:
        return 0, 1, total_tokens

def rank2symmetry(llm_interface, LLM_rank_args, cand1,cand2):
    '''
    Use llm to compare the height, due to the sequence, you need to compare each of the two in the front
    '''
    single_rank_func = LLM_rank_args["rank_func"]
    score = [0,0]
    bigger1,query_count1, total_tokens1 = single_rank_func(llm_interface, LLM_rank_args, cand1,cand2)
    score[1 - bigger1] += 1
    bigger2,query_count2, total_tokens2 = single_rank_func(llm_interface, LLM_rank_args, cand2,cand1)
    score[bigger2] += 1
    if score[0] > score[1]:
        return 1 , query_count1 + query_count2, total_tokens1 + total_tokens2
    elif score[0] < score[1]:
        return -1, query_count1 + query_count2, total_tokens1 + total_tokens2
    else:
        return 0, query_count1 + query_count2, total_tokens1 + total_tokens2

def sum_based_rankn(llm_interface,LLM_rank_args, candidates):
    '''
    All pairs are sorted pairwise, sum the total points, and choose the best
    '''
    total_querys = 0
    total_tokens = 0
    scores = [0]*len(candidates)
    for i in range(len(candidates)-1):
        for j in range(i+1,len(candidates)):
            pairwise_rank,query_count,rank2_tokens = rank2symmetry(llm_interface,LLM_rank_args, candidates[i],candidates[j])
            total_querys += query_count
            total_tokens += rank2_tokens
            if pairwise_rank > 0:
                scores[i] += 1
            elif pairwise_rank < 0:
                scores[j] += 1
            else:
                scores[i] += 0.5
                scores[j] += 0.5
    return scores, total_querys, total_tokens

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None):
    use_messages = []
    for message in messages:
        if not("valid" in message.keys() and message["valid"] == False):
            use_messages.append(message)
    json_data = {
        "model": model_name,
        "messages": use_messages,
    }
    if functions is not None:
        json_data.update({"functions": functions})
    
    try:
        openai_response = call_gpt(
            **json_data,
        )
        return openai_response 

    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"OpenAI calling Exception: {e}")
        return e

def get_clean_trajectory_with_action_args(final_node, query):
    global FINAL_TRAJECTORY, FINAL_FUNC_NAMES
    FINAL_TRAJECTORY = [f"Question:{query.strip()}"]
    FINAL_FUNC_NAMES = []

    node = final_node
    action_steps = []

    while node is not None:
        if node.node_type == "Action Input":
            parent = node.father
            if parent and parent.node_type == "Action":
                func_name = parent.description
                args = node.description
                func_call_str = f"functions.{func_name}{args}"
                action_steps.append(func_call_str)
                FINAL_FUNC_NAMES.append(func_name)
        node = node.father

    action_steps = list(reversed(action_steps))
    for i, step in enumerate(action_steps):
        FINAL_TRAJECTORY.append(f"Action {i+1}: {step}")
    
    return FINAL_TRAJECTORY, list(reversed(FINAL_FUNC_NAMES))

def add_experience(experience, query_id, env, api_list):
    # tool_order, used_api = get_action_name_trajectory(node)
    global FINAL_TRAJECTORY, FINAL_FUNC_NAMES
    print('FINAL_FUNC_NAMES:', FINAL_FUNC_NAMES)
    print(FINAL_TRAJECTORY)
    experience.execution_strategy = ExecutionStrategy(
        type="sequential",
        merge_logic="merge",
        tool_order=FINAL_TRAJECTORY
    )
    print(type(experience.execution_strategy))
    timestamp = time.time()
    readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    experience.metadata = Metadata(
        created_by="admin",
        created_at=str(readable_time),
        reuse_count=0,
        source_task_id=query_id
    )
    selected_api_list = []
    for item in FINAL_FUNC_NAMES:
        if item in env.functions_names:
            idx = env.functions_names.index(item)
            selected_api_list.append(api_list[idx])
    # experience.tool_details = [ToolDetail(**d) for d in selected_api_list]
    experience.tool_details = [
        ToolDetail(**{k: v for k, v in d.items() if k != 'description'}) 
        for d in selected_api_list
    ]
    print(experience)
    logger.info(f"[INFO] The experience was added successfully: {experience}")
    return experience

def dfs_search(args, query_data, api_list, i, tool_des, experience=None, messages=None):
    global FINAL_TRAJECTORY, FINAL_FUNC_NAMES, logger
    logger = logging.getLogger(f'task_{query_data["query_id"]}')
    dfs_runner = pipeline_runner()
    data_dict = {}
    result_data = {}
    data_dict['query'] = query_data['query']
    data_dict['api_list'] = api_list
    FINAL_TRAJECTORY = []
    FINAL_FUNC_NAMES = []
    task = ('DFS_woFilter_w2', data_dict, args.output_dir, tool_des)
    output = "Did not find the correct answer"
    env = ToolEnv(query=query_data['query'], api_list=api_list, tool_des=tool_des, process_id=i)
    final_exp = None

    for _ in range(3):
        result,tokens_len,output = dfs_runner.run(*task, messages, env)
        result_data = json.loads(result)
        if 'final_answer' in result_data and  result_data['final_answer'] != '':
            if(experience is not None):
                final_exp = add_experience(experience, query_data['query_id'], env, api_list)
            solved = True
        else:
            solved = False
        return solved, result_data,tokens_len,output,final_exp
    result_data = ''
    return False, result_data,tokens_len,output,final_exp
      
if __name__ == "__main__":
    args = parse_args()
    with open(args.query_path, 'r', encoding='utf-8') as f:
        query_data_all = json.load(f)
    query_data = query_data_all[25]
    api_list = [{'category_name': 'eCommerce', 'tool_name': 'Amazon Pricing and Product Info', 'api_name': 'Main Endpoint'}, {'category_name': 'Financial', 'tool_name': 'Currency Converter_v2', 'api_name': 'Convert'}]
    tools_des, to_prune = get_tools_des(api_list)
    dfs_search(args, query_data, api_list, tools_des, 25, experience=None)