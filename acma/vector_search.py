import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os 
import math
from utils import change_name, standardize

# ================= 配置区 =================
EMBEDDINGS_PATH = "./data/corpus_embeddings.pt" 
TSV_PATH = "./data/api_data.tsv" 
MODEL_PATH = "../../transformers/2025-07-24_02-49-05" 
tool_root_dir = '../../tools'

# ================= 执行区 =================
def search_similar_tools(query, top_k=3):
    print(f"正在查询: {query}")
    model = SentenceTransformer(MODEL_PATH)

    corpus_embeddings = torch.load(EMBEDDINGS_PATH)
    if isinstance(corpus_embeddings, torch.Tensor):
        corpus_embeddings = corpus_embeddings.cpu().numpy()
        
    # print("Loading metadata from TSV...")
    df = pd.read_csv(TSV_PATH, sep="\t", dtype=str)
    meta_list = []
    for _, row in df.iterrows():
        content_str = row.get("document_content", "")
        try:
            content = json.loads(content_str)
            content = json.loads(content)
        except json.JSONDecodeError:
            content = {}    

        meta_item = {
            "category_name": content.get("category", ""),
            "tool_name": content.get("tool", ""),
            "api_name": content.get("api", ""),
            "description": content.get("description", ""),
            "required_params": content.get("required_parameters", []),
        }
        meta_list.append(meta_item)
        
    corpus_size = corpus_embeddings.shape[0]
    if len(meta_list) != corpus_size:
        print(f"⚠️ 警告: 元数据数量 ({len(meta_list)}) 与 向量数量 ({corpus_size}) 不一致！")
    
    print("Encoding query...")
    query_embedding = model.encode([query])
    
    # 5. 计算相似度 
    sim_scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
    
    # 6. 找到分数最高的 Top K 个索引，argsort 会从小到大排序，所以用 [::-1] 反转变成从大到小
    top_indices = np.argsort(sim_scores)[::-1][:10]
    
    # 7. 整理结果
    results = []
    for idx in top_indices:
        sim_score = float(sim_scores[idx])
        
        tool_info = meta_list[idx] 
        stats = get_tool_info(tool_info)
        final_score = calculate_final_score(sim_score, stats)
        print("final_score: ",final_score)
        
        results.append({
            # "index": int(idx),
            "score": round(final_score, 4), 
            "category_name": tool_info.get("category_name"),
            "tool_name": tool_info.get("tool_name"),
            "api_name": tool_info.get("api_name"),
            "description": tool_info.get("description")
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top_result = results[:top_k]

    final_results = []
    for item in top_result:
        clean_item = {k: v for k, v in item.items() if k != 'score'}
        final_results.append(clean_item)
   
    return final_results


def get_tool_info(tool):
    cate_name = tool["category_name"]
    tool_name = standardize(tool["tool_name"])
    # api_name = change_name(standardize(tool["api_name"]))
    result = {
                "successRate": 1.0,
                "usageFrequency": 0,
                "avgExecutionTime": 1.0,
            }
    try:
        file_path = os.path.join(tool_root_dir, cate_name, tool_name + ".json")
        with open(file_path, "r", encoding="utf-8") as f:
            tool_json = json.load(f)
        for api in tool_json['api_list']:
            if api['name'] == tool["api_name"]:
                raw_score = api['score']
                result = {
                    "successRate": raw_score.get('successRate', 1.0),
                    "usageFrequency": raw_score.get('usageFrequency', 0),
                    "avgExecutionTime": raw_score.get('avgExecutionTime', 1.0),
                }
    except:
        print(tool_root_dir, cate_name, tool_name, "file is error")

    return result

def calculate_final_score(sim_score, stats):
    """
    计算包含响应速度的综合分数
    """
    success_rate = stats.get("successRate", 1.0)
    usage_freq = stats.get("usageFrequency", 0)
    exec_time = stats.get("avgExecutionTime", 1.0)
    
    # --- 权重设置 ---
    W_SIM  = 0.66
    W_RATE = 0.2
    W_FREQ = 0.05
    W_TIME = 0.1
    
    # --- 1. 相似度 ---
    sim_score_val = sim_score
    
    # --- 2. 成功率 ---
    rate_score_val = success_rate
    
    # --- 1. 处理频率 (归一化) ---
    # 使用 Log 函数平滑处理：log(1 + x)
    freq_score_val = math.log(1 + usage_freq) / math.log(101)
    
    # --- 4. 响应速度 ---
    # 设定：1秒以内得满分 1.0，超过 10 秒得 0 分，中间线性衰减。
    if exec_time <= 1.0:
        time_score_val = 1.0
    elif exec_time >= 10.0:
        time_score_val = 0.0
    else:
        time_score_val = 1.0 - (exec_time - 1.0) / 9.0
        
    final_score = (
        sim_score_val * W_SIM + 
        rate_score_val * W_RATE + 
        freq_score_val * W_FREQ + 
        time_score_val * W_TIME
    )
    
    return final_score

tool_json = get_tool_info({'category_name': 'SMS', 'tool_name': 'PhoneNumberValidate', 'api_name': 'Validate'})
print(tool_json)

