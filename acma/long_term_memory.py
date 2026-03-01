from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import os
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import uuid
from datetime import datetime
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import List, Dict, Any
from models import gpt_for_data
from prompt import INTENT_DECOMPOSER_PROMPT
from models import model_name
# ---------------------------
# 路径配置与初始化
# ---------------------------
DB_PATH = "./data/long_term_memory.db"
FAISS_INDEX_PATH = "./data/faiss_mpnet.index"
EMBEDDING_MODEL = "../../transformers/all-mpnet-base-v2"

model = SentenceTransformer(EMBEDDING_MODEL)
embedding_dim = 768

if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("[INFO] FAISS index loaded.")
else:
    base_index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap(base_index)  
    print("[INFO] New FAISS IDMap index created.")
print(f"FAISS index total entries: {index.ntotal}")

# ---------------------------
# 数据库连接
# ---------------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
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
conn.commit()


# ---------------------------
# 数据结构
# ---------------------------
@dataclass
class Subtask:
    intent: str
    api: str

    def to_text(self) -> str:
        return f"{self.intent} {self.api}"

@dataclass
class ToolDetail:
    category_name: str
    tool_name: str
    api_name: str

@dataclass
class ExecutionStrategy:
    type: str  
    merge_logic: str  
    tool_order: List[str]

    def to_dict(self):
        return asdict(self)

@dataclass
class Metadata:
    created_by: str
    created_at: str
    reuse_count: int
    source_task_id: str

@dataclass
class Experience:
    experience_id: str
    task_scene: str
    subtasks: List[Subtask]
    embedding: List[float] = field(default_factory=list)
    execution_strategy: ExecutionStrategy = None
    metadata: Metadata = None
    keywords: List[str] = field(default_factory=list)
    tool_details: List[ToolDetail] = field(default_factory=list)  
    faiss_index_id: int = None

    def generate_embedding(self):
        subtask_texts = " + ".join([s["intent"] for s in self.subtasks])
        vec = model.encode([subtask_texts])[0]
        self.embedding = (vec / np.linalg.norm(vec)).tolist()

    def to_sql_record(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "task_scene": self.task_scene,
            "subtasks_json": json.dumps(self.subtasks, ensure_ascii=False),
            "keywords_json": json.dumps(self.keywords, ensure_ascii=False),
            "embedding": json.dumps(self.embedding),
            "execution_strategy_json": json.dumps(self.execution_strategy.__dict__, ensure_ascii=False),
            "metadata_json": json.dumps(self.metadata.__dict__, ensure_ascii=False),
            "faiss_index_id": self.faiss_index_id,
            "tool_details_json": json.dumps([t.__dict__ for t in self.tool_details], ensure_ascii=False)  
        }

    def get_embedding_vector(self) -> List[float]:
        return self.embedding

    def __str__(self):
        return (
            f"Experience(\n"
            f"  ID: {self.experience_id}\n"
            f"  Scene: {self.task_scene}\n"
            f"  Subtasks: {self.subtasks}\n"
            f"  Embedding Dim: {len(self.embedding)}\n"
            f"  Keywords: {self.keywords}\n"
            f"  Execution Strategy: type={self.execution_strategy.type}, "
            f"merge_logic={self.execution_strategy.merge_logic}, "
            f"tool_order={self.execution_strategy.tool_order}\n"
            f"  Metadata: created_by={self.metadata.created_by}, "
            f"created_at={self.metadata.created_at}, "
            f"reuse_count={self.metadata.reuse_count}, "
            f"source_task_id={self.metadata.source_task_id}\n"
            f"  Tool Details: {self.tool_details}\n"
            f"  FAISS Index ID: {self.faiss_index_id}\n"
            f")"
        )

    @staticmethod
    def from_sql_record(record: Dict[str, Any]) -> 'Experience':
        subtasks_data = json.loads(record["subtasks_json"])
        subtasks = [Subtask(**s) for s in subtasks_data]

        execution_strategy = None
        if record.get("execution_strategy_json"):
            execution_strategy_data = json.loads(record["execution_strategy_json"])
            execution_strategy = ExecutionStrategy(**execution_strategy_data)

        metadata = None
        if record.get("metadata_json"):
            metadata_data = json.loads(record["metadata_json"])
            metadata = Metadata(**metadata_data)

        keywords = json.loads(record["keywords_json"])
        # embedding = json.loads(record["embedding"])

        tool_details = []
        if record.get("tool_details_json"):
            tool_details_data = json.loads(record["tool_details_json"])
            tool_details = [ToolDetail(**t) for t in tool_details_data]

        return Experience(
            experience_id=record["experience_id"],
            task_scene=record["task_scene"],
            subtasks=subtasks,
            embedding=[],  # 不从数据库加载 embedding，直接赋空
            execution_strategy=execution_strategy,
            metadata=metadata,
            keywords=keywords,
            faiss_index_id=record.get("faiss_index_id"),
            tool_details=tool_details
        )


# ---------------------------
# 工具函数
# ---------------------------
def is_duplicate_experience(task_scene: str, subtasks: List[Subtask]) -> bool:
    cursor.execute("SELECT task_scene, subtasks_json FROM memories")
    rows = cursor.fetchall()
    new_set = set((s["intent"], str(s["api"])) for s in subtasks)

    for row in rows:
        existing_scene = row[0]
        existing_subtasks = json.loads(row[1])
        existing_set = set((s['intent'], str(s['api'])) for s in existing_subtasks)

        if task_scene == existing_scene and new_set == existing_set:
            return True
    return False

def extract_core_info(similar):
    if not similar:
        print("[INFO] No similar experience was found")
        return None

    best_sim, best_score = similar[0]

    intents = [s.intent for s in best_sim.subtasks]
    strategy = best_sim.execution_strategy
    tool_details = []
    seen = set()
    for t in best_sim.tool_details:
        t_dict = t.__dict__
        t_key = tuple(sorted(t_dict.items()))
        if t_key not in seen:
            seen.add(t_key)
            tool_details.append(t_dict)

    result = {
        "experience_id": best_sim.experience_id,
        "task_scene": best_sim.task_scene,
        "score": round(float(best_score), 4),
        "subtask_intents": intents,
        # "trajectory": strategy,
        # "apis": apis,
        "execution_type": strategy.type,
        "merge_logic": strategy.merge_logic,
        "execution_strategy": strategy.tool_order,
        "tool_details": tool_details,  
        "reuse_count": best_sim.metadata.reuse_count
    }

    return result

def change_faiss_format():
    """
    将数据库中的数据转换为 IndexIDMap 格式
    """
    FAISS_INDEX_PATH = "faiss_mpnet.index"     
    SQLITE_DB_PATH = "long_term_memory.db"       

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT faiss_index_id, embedding FROM memories ORDER BY faiss_index_id ASC")
    rows = cursor.fetchall()

    if not rows:
        print("[ERROR] 数据库中没有记录")
        exit()

    ids = []
    embeddings = []
    for row in rows:
        ids.append(int(row[0]))
        embeddings.append(np.array(json.loads(row[1]), dtype=np.float32))

    embedding_dim = len(embeddings[0])
    base_index = faiss.IndexFlatIP(embedding_dim)
    new_index = faiss.IndexIDMap(base_index)

    embeddings_np = np.array(embeddings).astype("float32")
    ids_np = np.array(ids, dtype=np.int64)

    new_index.add_with_ids(embeddings_np, ids_np)
    faiss.write_index(new_index, FAISS_INDEX_PATH)

    print(f"[DONE] 已写入 IndexIDMap 索引，共 {new_index.ntotal} 条数据")

    conn.close()

def update_faiss_index():
    """ 
    更新 FAISS 索引
    """
    base_index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap(base_index)

    cursor.execute("SELECT embedding, faiss_index_id FROM memories")
    for emb_str, faiss_id in cursor.fetchall():
        vec = np.array(json.loads(emb_str), dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        index.add_with_ids(np.array([vec]), np.array([faiss_id]))

    faiss.write_index(index, FAISS_INDEX_PATH)


# ---------------------------
# 添加经验（自动持久化）
# ---------------------------
def add_experience_to_db(exp: Experience):
    if is_duplicate_experience(exp.task_scene, exp.subtasks):
        print("[SKIP] Repeated experience, not saved.")
        return

    exp.generate_embedding()
    exp.faiss_index_id = index.ntotal 

    index.add_with_ids(
        np.array([exp.embedding]).astype("float32"),
        np.array([exp.faiss_index_id], dtype=np.int64)
    )

    faiss.write_index(index, FAISS_INDEX_PATH)

    record = exp.to_sql_record()

    cursor.execute('''
        INSERT INTO memories 
        (experience_id, task_scene, subtasks_json, keywords_json, embedding, execution_strategy_json, metadata_json, faiss_index_id, tool_details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record["experience_id"],
        record["task_scene"],
        record["subtasks_json"],
        record["keywords_json"],
        record["embedding"],
        record["execution_strategy_json"],
        record["metadata_json"],
        record["faiss_index_id"],
        record["tool_details_json"]
    ))

    conn.commit()
    print(f"[ADD] The experience {exp.experience_id} was added successfully.")

    exp_for_json = asdict(exp)
    exp_for_json["embedding"] = None
    exp_for_json["subtasks"] = exp.subtasks

    EXPERIENCE_JSON_PATH = "./data/experiences.json"

    if os.path.exists(EXPERIENCE_JSON_PATH):
        with open(EXPERIENCE_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(exp_for_json)

    with open(EXPERIENCE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"[EXPORT] Experience saved to {EXPERIENCE_JSON_PATH}")


# ---------------------------
# 检索经验
# ---------------------------
def search_similar_by_vector(query_vec: np.ndarray, top_k: int = 3, threshold=0.8) -> List[tuple]:
    query_vec = (query_vec / np.linalg.norm(query_vec)).astype("float32")
    D, I = index.search(np.array([query_vec]), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1 or score < threshold:
            continue
        cursor.execute("SELECT * FROM memories WHERE faiss_index_id = ?", (int(idx),))
        row = cursor.fetchone()
        if row:
            exp = Experience(
                experience_id=row[0],
                task_scene=row[1],
                subtasks=[Subtask(**s) for s in json.loads(row[2])],
                keywords=json.loads(row[3]),
                embedding=json.loads(row[4]),
                execution_strategy=ExecutionStrategy(**json.loads(row[5])),
                metadata=Metadata(**json.loads(row[6])),
                faiss_index_id=row[7],  
                tool_details=[ToolDetail(**t) for t in json.loads(row[8])]
            )
            results.append((exp, score))
    return results

def semantic_rerank_by_intents(similar: List[tuple], current_intents: List[str]) -> List[tuple]:
    """
    分别编码每一个子意图 → 得到一个列表向量 sub_vecs
    用句向量余弦相似度，重新把相似经验按和当前用户意图的 语义匹配程度 排序。
    """
    current_vecs = model.encode(current_intents, convert_to_tensor=True)

    def compute_semantic_score(exp: Experience) -> float:
        sub_intents = [s.intent for s in exp.subtasks]
        if not sub_intents:
            return 0.0
        sub_vecs = model.encode(sub_intents, convert_to_tensor=True)

        score_matrix = util.cos_sim(current_vecs, sub_vecs)  # shape: [len(cur) x len(sub)]
        max_scores = score_matrix.max(dim=1).values  
        avg_score = max_scores.mean().item()
        return avg_score

    sorted_similar = sorted(similar, key=lambda x: compute_semantic_score(x[0]), reverse=True)
    return sorted_similar


# ---------------------------
# 更新经验被重用次数
# ---------------------------
def update_reuse_count_in_db(experience_id: str):
    cursor.execute('SELECT metadata_json FROM memories WHERE experience_id = ?', (experience_id,))
    result = cursor.fetchone()

    if not result:
        print(f"[ERROR] No experience found with ID {experience_id}")
        return

    metadata_json = result[0]
    metadata = json.loads(metadata_json)
    metadata['reuse_count'] = int(metadata['reuse_count']) + 1

    cursor.execute('''
        UPDATE memories 
        SET metadata_json = ?
        WHERE experience_id = ?
    ''', (json.dumps(metadata, ensure_ascii=False), experience_id))

    conn.commit()

    print(f"[UPDATE] reuse_count updated for experience {experience_id}")


# ---------------------------
# 删除经验
# ---------------------------
def delete_experience(experience_id: str):
    cursor.execute("SELECT faiss_index_id FROM memories WHERE experience_id = ?", (experience_id,))
    result = cursor.fetchone()
    
    if not result:
        print(f"[WARN] Experience ID {experience_id} not found.")
        return
    
    faiss_id = result[0]
    print(f"[INFO] Deleting experience_id={experience_id}, faiss_index_id={faiss_id}")

    global index  
    index.remove_ids(faiss.IDSelectorRange(faiss_id, faiss_id + 1))  
    
    faiss.write_index(index, FAISS_INDEX_PATH)  

    cursor.execute("DELETE FROM memories WHERE experience_id = ?", (experience_id,))
    conn.commit()

    print(f"[DONE] Experience {experience_id} deleted from database and FAISS index.")


# ---------------------------
# 对外接口
# ---------------------------
def get_similar_experiences(query: str, top_k: int = 3, threshold=0.5) -> Dict:
    prompt = INTENT_DECOMPOSER_PROMPT.replace('{USER_INPUT}', query)
    response = gpt_for_data(model=model_name, prompt=prompt, temperature=0.0)
    print(response.choices[0].message.content)
    content_str = json.loads(response.choices[0].message.content)
    task_list = content_str.get("task_list", "")
    task_scene = content_str.get("task_scene", "")
    subtasks = [Subtask(intent=task["subtask"], api="") for task in task_list]
    task_list_str = ' + '.join([task["subtask"] for task in task_list])
    print("task_list_str:", task_list_str)
    vec = model.encode(task_list_str)
    embedding_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
    print(f"Parsed task_list: {task_list}")
    
    similars = search_similar_by_vector(vec, top_k=top_k, threshold=threshold)
    
    for sim, score in similars:
        print(f"[FOUND] ID: {sim.experience_id}, Scene: {sim.task_scene}, Score: {score:.4f}, Subtasks: {[s.intent for s in sim.subtasks]}, Keywords: {sim.keywords}")

    intents = [task["subtask"] for task in task_list]
    similar = semantic_rerank_by_intents(similars, intents)
    message = extract_core_info(similar)

    new_exp = Experience(
        experience_id=str(uuid.uuid4()),
        task_scene=task_scene,
        subtasks=subtasks,
        embedding=embedding_list,
        keywords=[],
        execution_strategy=ExecutionStrategy(type="", merge_logic="", tool_order=[]),
        metadata=Metadata(created_by="", created_at="", reuse_count=0, source_task_id=""),
        tool_details=[],
        faiss_index_id=None
    )

    return message, new_exp

# ---------------------------
# 示例运行
# ---------------------------
if __name__ == "__main__":
    query = "I'm currently researching the drug Metformin to understand its generic name and check its price history. Can you also help me find recent articles or studies related to it?"
    # qts=time.time()
    get_similar_experiences(query)
    # get_similar_experiences(query)
    # get_similar_experiences(query)
    # get_similar_experiences(query)
    # print(time.time()-qts)
    # print(time.time()-ts)
    example = Experience(
        experience_id=str(uuid.uuid4()),
        task_scene="Stock News and International Money Transfer",
        subtasks=[
            Subtask(
                intent="Get the latest stock market news for Tesla",
                api="Stock News"
            ),
            Subtask(
                intent="Make a money transfer to family in India",
                api="Authentic Money Transfer Portal"
            )
        ],
        tool_details=[
            ToolDetail(
                category_name="Finance",
                tool_name="Real-Time Finance Data",
                api_name="Stock News"
            ),
            ToolDetail(
                category_name="Payments",
                tool_name="fastmoney",
                api_name="Authentic Money Transfer Portal"
            )
        ],
        execution_strategy=ExecutionStrategy(
            type="Sequential Execution",
            merge_logic="First fetch stock news, then process money transfer",
            tool_order=["Real-Time Finance Data", "fastmoney"]
        ),
        metadata=Metadata(
            created_by="user",
            created_at=str(datetime.now().date()),
            reuse_count=0,
            source_task_id="task-00016"
        ),
        keywords=["stock", "news", "money transfer", "Tesla", "India"]
    )
    add_experience_to_db(example)
    


