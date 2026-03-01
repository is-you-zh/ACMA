from prompt import RETRIEVAL_PROMPT, INTENT_DECOMPOSER_PROMPT, CHECK_TASK_COMPLETION_PROMPT, RESUME_PROMPT
from termcolor import colored
from models import *
from function_database import *
from l1_cache_manager import L1CacheManager
from l2_cache_manager import L2CacheManager
from vector_search import search_similar_tools

class Retrieval_Agent:
    def __init__(self, query, logger, subtask):
        self.logger = logger
        self.query = query  
        self.api_pool = []
        self.total_tokens = 0
        self.stop = False
        self.l1_cache_manager = None
        self.l2_cache_manager = None
        
        self.api_mapping = {
            "decompose_task": self.decompose_task,
            "search_apis": self.search_apis,
            "delete_api": self.delete_api,
            "check_task_completion": self.check_task_completion,
            "finish_func": self.finish_func
        }

        self.functions = [
            decompose_task_function,
            search_apis_function,
            # delete_api_function,
            check_task_completion_function,
            finish_function
        ]

        if subtask:
            self.pending_subtasks = subtask
            print(colored(f'Starting tool retrieval for: {self.pending_subtasks}', 'blue'))
            self.logger.info(f"Starting tool retrieval for: {self.pending_subtasks}")
        else:
            self.pending_subtasks = self.decompose_task(self.query)
            print(colored(f'Starting tool retrieval for: {self.pending_subtasks}', 'blue'))
            self.logger.info(f"Starting tool retrieval for: {self.pending_subtasks}")

        self.messages = [
            {
                "role": "system",
                "content": RETRIEVAL_PROMPT
            },
            {   "role": "user",
                "content": f"Please determine relevant tools and assign them. Task description: {query}, the sub-tasks that have not been completed yet are: {self.pending_subtasks} Begin!"
            }
        ]
        logger.info(f'RetrievalAgent prompt: {self.messages}')

    def decompose_task(self, query):
        """
        使用大模型将任务分解成多个子任务。模型根据任务描述自动分解。
        """

        prompt = INTENT_DECOMPOSER_PROMPT.replace('{USER_INPUT}', query)
        response = gpt_for_data(model=model_name, prompt=prompt, temperature=0.1)

        content_str = json.loads(response.choices[0].message.content)
        subtasks = content_str.get("task_list", "")
        self.logger.info(f"Decomposed tasks: {subtasks}")
        return subtasks

    def search_apis(self, subquery):
        """
        根据子任务描述在工具池中检索工具。
        使用缓存查询工具。如果没有命中缓存，则执行常规检索，并更新缓存。
        """
        subquery = json.loads(subquery)
        # 查询L1缓存
        if not self.l1_cache_manager:
            self.l1_cache_manager = L1CacheManager(api_lib_path=None, max_cache_size=100)
        l1_cached_result = self.l1_cache_manager.lookup(subquery["subtask"])
        if l1_cached_result:
            self.logger.info(f"L1 Cache hit for subtask: {subquery}")
            if isinstance(l1_cached_result, str):
                l1_cached_result = json.loads(l1_cached_result)
            return l1_cached_result

        # 查询L2缓存
        if not self.l2_cache_manager:
            self.l2_cache_manager = L2CacheManager()
        l2_cached_result = self.l2_cache_manager.lookup(subquery["subtask"],subquery["sublabel"]) 
        if l2_cached_result:
            self.logger.info(f"L2 Cache hit for subtask: {subquery}")
            if isinstance(l2_cached_result, str):
                l2_cached_result = json.loads(l2_cached_result)
            return l2_cached_result  
        
        # 如果缓存没有命中，进行常规的工具检索，相似度检索
        matching_tools = search_similar_tools(subquery["subtask"])

        return matching_tools

    def check_task_completion(self):
        """
        检查是否所有子任务都已经完成。
        """
        prompt = CHECK_TASK_COMPLETION_PROMPT.replace('{task}', str(self.pending_subtasks)).replace('{tools}', str(self.api_pool))
        response = gpt_for_data(model=model_name, prompt=prompt, temperature=0.0)
        result = json.loads(response.choices[0].message.content)
        print(result)

        return result

    def execute(self):
        """
        主执行方法，分解任务、检索工具并判断完成情况。
        模仿工具搜索的逻辑，根据任务进展反馈工具分配和处理。
        """

        for i in range(11):
            if i==10 or self.stop or self.total_tokens > 200000:
                self.logger.info(f"stop: {self.stop}")
                self.logger.info(f"total_tokens: {self.total_tokens}")
                print('#'*100)
                print(colored('stop', 'red'))
                print(colored(f'total token: {self.total_tokens}','red'))
                return self.api_pool, self.total_tokens

            try:
                response = call_gpt(
                    messages=self.messages,
                    model=model_name,
                    functions=self.functions
                )
                # self.logger.info(f"Messages:\n{json.dumps(self.messages, indent=4, ensure_ascii=False)}")
            except Exception as e:  
                self.stop = True
                self.logger.info(f"Error detected: {e}")      
                continue        
            if isinstance(response, str):
                continue
            self.total_tokens += response.usage.total_tokens
            print('#'*100)
            print(colored(f'The {i+1} loop of api_search:','yellow'))
            if(response.choices[0].message.content):
                print('Thought:', response.choices[0].message.content)
                message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content if response.choices[0].message.content is not None else '',
                }
                self.messages.append(message)
                self.logger.info(message)

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                print('tool call number', len(tool_calls), tool_calls)
                self.logger.info(f"tool call number: {len(tool_calls)}")
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    if function_name not in self.api_mapping:
                        function_name = 'hullucinating_function_name'
                        function_call_result = "Function name error"
                        self.logger.info("hullucinating_function_name")

                    elif function_name == 'decompose_task':
                        function_call_result = self.decompose_task(function_args)

                    elif function_name == 'search_apis':
                        try:
                            function_call_result = self.search_apis(function_args)
                            for item in function_call_result:
                                if item not in self.api_pool:
                                    self.api_pool.append(item)
                            function_call_result = str(function_call_result) + f"substak {function_args} has been completed."

                        except Exception as e:
                            print(e, function_name, function_args, file=open(f'{args.output_dir}/error.txt', 'a', encoding='utf-8'))
                            function_call_result = 'input format error'
                        print("function_call_result: ",function_call_result)

                    elif function_name == 'check_task_completion':
                        solvable = False
                        function_call_result = self.check_task_completion()   
                        solvable = function_call_result['solvable']
                        reason = function_call_result['reason']
                        print(colored(f"'solvable: ', {solvable}, ' ,reason: ', {reason}", "yellow"))
                        self.logger.info(function_call_result)
                        if solvable:
                            self.finish_func()
                            return self.api_pool, self.total_tokens

                    elif function_name == 'finish_func':
                        self.finish_func()
                        return self.api_pool, self.total_tokens                          

                    message = {
                                "role":"assistant",
                                "function_name": function_name,
                                "function_param": function_args,
                                "content": str(function_call_result),
                            }
                    self.messages.append(message)
                    print(json.dumps(message, ensure_ascii=False, indent=2))
                    self.logger.info(json.dumps(message, ensure_ascii=False, indent=2))  
            else:
                message = {
                    'role': "user",
                    'content': 'Please strictly follow the list of tools I provided for you to invoke. Do not just think about it but fail to act.'
                }
                self.messages.append(message)
                self.logger.info(message)        

        return self.api_pool, self.total_tokens
    
    def delete_api(self, apis_to_remove):
        for api in apis_to_remove:
            print(colored(f'removing api: {api}', 'red'))
            self.api_pool.remove(api)

    def resume_search(self, fail_reason, apis_to_remove):
        print(colored('Refind Begin', 'red'))
        self.logger.info('Refind Begin')
        self.stop = False
        resume_tokens = 0
        prompt = RESUME_PROMPT.replace('{query}', self.query).replace('{api_pool}', str(self.api_pool)).replace('{fail_reason}',fail_reason)
        if apis_to_remove:
            self.delete_api(apis_to_remove)
            prompt += f" Deleted API: {apis_to_remove}"
        self.logger.info(f"Resume prompt: {prompt}")
        response = gpt_for_data(model=model_name, prompt=prompt, temperature=1.0)
        resume_tokens += response.usage.total_tokens
        content_str = json.loads(response.choices[0].message.content)
        self.pending_subtasks = content_str.get("task_list", "")
        self.logger.info(f"Resume decomposed tasks: {self.pending_subtasks}")
        message = {
            'role': "user",
            'content': f'Execution failed, reason: {fail_reason}, pending tasks to resolve: {self.pending_subtasks}, available APIs: {self.api_pool}'
        }
        self.messages.append(message)
        self.logger.info(message) 
        
        for i in range(6):
            if i==6 or self.stop:
                self.logger.info(f"stop: {self.stop}")
                print('#'*100)
                print(colored('stop', 'red'))
                return self.api_pool, resume_tokens

            try:
                response = call_gpt(
                    messages=self.messages,
                    model=model_name,
                    functions=self.functions
                )
            except Exception as e:  
                self.stop = True
                self.logger.info(f"Error detected: {e}")      
                continue        
            if isinstance(response, str):
                continue
            resume_tokens += response.usage.total_tokens
            print('#'*100)
            print(colored(f'The {i+1} loop of resume:','yellow'))
            if(response.choices[0].message.content):
                print('Thought:', response.choices[0].message.content)
                message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content if response.choices[0].message.content is not None else '',
                }
                self.messages.append(message)
                self.logger.info(message)

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                print('tool call number', len(tool_calls), tool_calls)
                self.logger.info(f"tool call number: {len(tool_calls)}")
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    if function_name not in self.api_mapping:
                        function_name = 'hullucinating_function_name'
                        function_call_result = "Function name error"
                        self.logger.info("hullucinating_function_name")

                    elif function_name == 'decompose_task':
                        function_call_result = self.decompose_task(function_args)

                    elif function_name == 'search_apis':
                        try:
                            function_call_result = self.search_apis(function_args)
                            for item in function_call_result:
                                if item not in self.api_pool:
                                    self.api_pool.append(item)
                            function_call_result = str(function_call_result) + f"substak {function_args} has been completed."

                        except Exception as e:
                            print(e, function_name, function_args, file=open(f'{args.output_dir}/error.txt', 'a', encoding='utf-8'))
                            function_call_result = 'input format error'
                        print("function_call_result: ",function_call_result)

                    elif function_name == 'check_task_completion':
                        solvable = False
                        function_call_result = self.check_task_completion()   
                        solvable = function_call_result['solvable']
                        reason = function_call_result['reason']
                        print(colored(f"'solvable: ', {solvable}, ' ,reason: ', {reason}", "yellow"))
                        self.logger.info(function_call_result)
                        if solvable:
                            self.finish_func()
                            return self.api_pool, resume_tokens

                    elif function_name == 'finish_func':
                        self.finish_func()
                        return self.api_pool, resume_tokens                       

                    message = {
                                "role":"assistant",
                                "function_name": function_name,
                                "function_param": function_args,
                                "content": str(function_call_result),
                            }
                    self.messages.append(message)
                    print(json.dumps(message, ensure_ascii=False, indent=2))
                    self.logger.info(json.dumps(message, ensure_ascii=False, indent=2))  
            else:
                message = {
                    'role': "user",
                    'content': 'Please strictly follow the list of tools I provided for you to invoke. Do not just think about it but fail to act.'
                }
                self.messages.append(message)
                self.logger.info(message)        

        return self.api_pool, resume_tokens

    def finish_func(self):
        """
        结束任务的处理。
        """
        print(f'apis {self.api_pool} assigned')
        self.logger.info(f'apis {self.api_pool} assigned')
        print(colored('tool finish search', 'green'))
        self.logger.info('tool finish search')
        message = {
                    "role":"assistant",
                    "function_name": "finish_func",
                    "content": 'Finished'
                }
        print(json.dumps(message, ensure_ascii=False, indent=2))
        self.logger.info(json.dumps(message, ensure_ascii=False, indent=2))   


    # def find_apis(self):
    #     # 遍历待处理的子任务
    #     for i in range(len(self.pending_subtasks)):
    #         subtask = self.pending_subtasks[i]
    #         print(f"Processing subtask {i+1}/{len(self.pending_subtasks)}: {subtask}")
    #         self.logger.info(f"Processing subtask {i+1}/{len(self.pending_subtasks)}: {subtask}")
            
    #         # 检索工具并选择最合适的工具
    #         tools = self.search_tools(subtask)
    #         chosen_tool = self.choose_tool(tools)

    #         if chosen_tool:
    #             self.logger.info(f"Selected tool: {chosen_tool['name']} for subtask '{subtask}'")
    #             # 假设工具调用是执行子任务的关键
    #             self.completed_subtasks.append(subtask)  # 标记子任务完成
    #             self.pending_subtasks.remove(subtask)  # 从待处理子任务中移除
    #         else:
    #             # 如果没有找到合适的工具，通知用户并将子任务重新加入待处理队列
    #             self.logger.info(f"No suitable tool found for subtask '{subtask}'. Reattempting.")
    #             self.pending_subtasks.append(subtask)  # 将未完成的子任务重新添加到队列
            
    #         # 检查是否所有子任务都已完成
    #         if self.check_task_completion():
    #             self.logger.info("All subtasks completed successfully.")
    #             break
            
    #         # 状态检查：例如，如果超出 token 限制，或触发了某些终止条件，则停止执行
    #         if self.stop or self.total_tokens > 200000:
    #             self.logger.info(f"Execution stopped due to token limit or other conditions.")
    #             print('#' * 100)
    #             print(colored('Execution stopped', 'red'))
    #             return f"Execution stopped. Total tokens: {self.total_tokens}"

    #         # 错误处理：如果发生异常，继续尝试
    #         if error_flag:
    #             self.logger.info("Error detected, continuing execution.")
    #             error_flag = False
    #             continue
            
    #         # 模拟类似于工具分配的过程，添加对应的反馈消息
    #         if self.finish_search:
    #             print(f"Tools assigned successfully for: {self.api_pool}")
    #             self.logger.info(f"Tools assigned successfully for: {self.api_pool}")
    #             self.finish_search = True
    #             return f"Tools assigned. Current task status: {self.status}"

    #         # 模拟与大模型进行交互，并处理反馈
    #         try:
    #             response = call_gpt(
    #                 messages=self.messages,
    #                 model=retrieval_model_name,
    #                 functions=self.functions
    #             )
    #         except Exception as e:
    #             self.logger.error(f"Error during GPT call: {str(e)}")
    #             continue

    #         if isinstance(response, str):
    #             continue
            
    #         self.total_tokens += response.usage.total_tokens
            
    #         # 输出 GPT 的反馈内容
    #         print('#' * 100)
    #         print(colored(f'GPT Response: {response.choices[0].message.content}', 'yellow'))
            
    #         # 处理工具调用的返回值
    #         tool_calls = response.choices[0].message.tool_calls
    #         if tool_calls:
    #             for tool_call in tool_calls:
    #                 function_name = tool_call.function.name
    #                 function_args = tool_call.function.arguments
                    
    #                 if function_name == 'finish':
    #                     self.logger.info('Finish task received.')
    #                     self.finish_search = True
    #                     self.messages.append({
    #                         "role": "function",
    #                         "name": function_name,
    #                         "content": 'Finished'
    #                     })
    #                     print(f"Task finished. Tools assigned: {self.api_pool}")
    #                     return f"Tools assigned. Final status: {self.status}"

    #                 elif function_name in self.api_mapping:
    #                     function_call_result = self.api_mapping[function_name](**json.loads(function_args))
    #                     message = {
    #                         "role": "function",
    #                         "name": function_name,
    #                         "param": function_args,
    #                         "content": str(function_call_result)
    #                     }
    #                     self.messages.append(message)
    #                     self.logger.info(f"Function call result: {json.dumps(message, ensure_ascii=False, indent=2)}")
    #                 else:
    #                     self.logger.warning(f"Unrecognized function: {function_name}")
    #                     message = {
    #                         "role": "function",
    #                         "name": function_name,
    #                         "content": "Function name error"
    #                     }
    #                     self.messages.append(message)

    #         # 最终输出反馈
    #         print(f"Tools assigned for: {self.api_pool}")
    #         self.logger.info(f"Tools assigned for: {self.api_pool}")
    #         self.finish_search = True
    #         return f"Tools assigned. Status: {self.status}"
    #         print(f'api_pool {self.api_pool} assigned')
    #         logger.info(f'api_pool {self.api_pool} assigned')
    #         self.finish_search = True
    #         if multi_thread:
    #             sem.release()
    #             return f'api_pool {self.api_pool} assigned'
    #         else:
    #             return f'api_pool {self.api_pool} assigned. The status of current found apis is: {status}'
            
            