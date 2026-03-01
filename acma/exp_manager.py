import json
import os
import re
from typing import List, Dict, Any
from typing import Optional, List
from long_term_memory import add_experience_to_db, Subtask
import time
from prompt import COMPLETE_SUBTASK_PROMPT, ALIASES_PROMPT
from models import gpt_for_data
from utils import get_api_details

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

class ExperienceManagementAgent:
    def __init__(self, final_exp,logger, max_cache_size: Optional[int] = 500):
        """
        初始化经验管理Agent，加载场景工具数据库和任务工具缓存文件。
        
        :param scene_tools_db: 场景级工具的SQLite数据库路径
        :param task_tools_file: 任务级工具的存储路径（JSON文件）
        """
        self.scene_tools_db = "./data/long_term_memory.db"
        self.task_tools_file = "./data/l1_cache.json"
        self.max_cache_size = max_cache_size
        self.task_tools = self._load_json(self.task_tools_file)
        self.exp = final_exp
        self.logger = logger

        # self.conn = sqlite3.connect(self.scene_tools_db)
        # self.cursor = self.conn.cursor()
        # self._initialize_scene_tools_table()

    def _load_json(self, file_path: str) -> Dict:
        """
        从指定路径加载JSON文件。
        """
        if not os.path.exists(file_path):
            return []  
        if os.path.getsize(file_path) == 0: 
            return [] 
        with open(file_path, 'r') as file:
            return json.load(file)
        
    def _initialize_scene_tools_table(self):
        """
        初始化场景工具表，如果表不存在则创建。
        """
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                experience_id TEXT PRIMARY KEY,
                task_scene TEXT,
                subtasks_json TEXT,
                keywords_json TEXT,
                embedding TEXT,
                execution_strategy_json TEXT,
                metadata_json TEXT,
                faiss_index_id INTEGER,
                tool_details_json TEXT
            )
        ''')
        self.conn.commit()

    def _save_json(self, data: Dict, file_path: str):
        """
        将数据保存到JSON文件。
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def add_task_tool(self, tool: str, intent: str, aliases: Optional[List[str]] = None):
        """
        向一级缓存添加一个新的条目。

        :param tool: 工具名称，例如 'WeatherAPI.get_current_weather'
        :param desc: 工具描述，例如 'Retrieve the current weather for a given city'
        :param example_queries: 可选的示例查询列表
        :param aliases: 可选的别名列表
        :return: 新添加的条目
        """
        task_tool = {
            "task_signature": {"tool": tool},
            "stats": {"usage": 0, "success": 0, "last_used": int(time.time())},
            "aliases": aliases or [],  
            "_key_norm": normalize_text(intent) 
        }

        # if example_queries:
        #     task_tool["example_queries"] = example_queries 

        self.task_tools.append(task_tool)

        # self._build_tfidf_index() 
        if self.max_cache_size is not None and len(self.task_tools) > self.max_cache_size:
            self.task_tools.sort(key=lambda e: e["stats"].get("last_used", 0))
            self.task_tools = self.task_tools[-self.max_cache_size:]

        self._save_json(self.task_tools, self.task_tools_file)
        
        return task_tool
    
    def update_database(self):
        prompt = COMPLETE_SUBTASK_PROMPT.replace('{experience}', str(self.exp))
        self.logger.info(f"prompt: {prompt}")
        response = gpt_for_data(prompt, temperature=0.1)
        data = json.loads(response.choices[0].message.content)
        print("complete subtask data:", data)
        self.logger.info(f"complete subtask data: {data}")
        self.exp.subtasks = data["subtasks"]
        print(self.exp)
        
        add_experience_to_db(self.exp)
        for subtask in data.get("subtasks", []):
            api_content = subtask.get("api")
            if not api_content:
                continue

            if isinstance(api_content, dict):
                api_list = [api_content]
            elif isinstance(api_content, list):
                api_list = api_content
            else:
                continue

            for api_obj in api_list:
                api_details = get_api_details(**api_obj)
                api_obj['description'] = api_details['description'] if 'description' in api_details else ''

            prompt = ALIASES_PROMPT.replace('{input}',str(subtask))
            response_for_aliases = gpt_for_data(prompt, temperature=1.0)
            response_content = response_for_aliases.choices[0].message.content
            # Convert string to list
            response_list = json.loads(response_content)
            self.add_task_tool(subtask["api"],subtask["intent"],response_list)

def get_current_timestamp():
    """获取当前时间戳（秒）"""
    return int(time.time())

def initialize_files_in_folder(folder_path):
    """初始化文件夹中所有 JSON 文件的 API 数据"""
    timestamp = get_current_timestamp()

    for folder_name in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder_name)

        if os.path.isdir(folder_path_full) and folder_name != 'server':
            for file_name in os.listdir(folder_path_full):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path_full, file_name)

                    try:
                        with open(file_path, 'r') as f:
                            tool_data = json.load(f)

                        if 'api_list' in tool_data:
                            for api in tool_data['api_list']:
                                if 'score' not in api:
                                    api['score'] = {
                                        'successRate': 1.0,  
                                        'usageFrequency': 0,  
                                        'successCount': 0,    
                                        'avgExecutionTime': 1.0,  
                                        'lastUpdated': timestamp,  
                                        '__typename': 'Score'  
                                    }

                        with open(file_path, 'w') as f:
                            json.dump(tool_data, f, indent=4)

                    except json.JSONDecodeError:
                        print(f"跳过文件 {file_path}：JSON 格式错误或为空。")
                    except Exception as e:
                        print(f"处理文件 {file_path} 时发生错误: {e}")

                   