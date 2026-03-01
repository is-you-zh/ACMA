import math
from typing import List, Dict, Optional
from long_term_memory import Experience

def count_connect_words(text: str) -> int:
    """
    统计文本中表示“多动作/多结果”的连接词数量（连接词越多→复杂度越高）
    已包含所有高频连接词：和/以及/与/同/并/然后/接着/同时/还/又/此外/加之 + 英文等效词
    """
    connect_words = [
        '和', '以及', '与', '同', '跟', '及',
        '并', '然后', '接着', '继而', '随后',
        '同时', '还', '又', '此外', '加之', '另外',
        'and', 'then', 'next', 'also', 'as well as', 'and also', 
        'in addition', 'plus', 'together with', 'along with'
    ]
    text_processed = text.lower().strip() 
    connect_count = 0
    for word in connect_words:
        connect_count += text_processed.count(word)
    return max(connect_count, 0)

def normalize_feature(value: int, min_val: int, max_val: int) -> float:
    """Min-Max归一化到[0,1]，处理边界值避免异常"""
    if max_val == min_val:
        return 0.0
    clamped_val = min(max(value, min_val), max_val)
    return (clamped_val - min_val) / (max_val - min_val)

def complexity_score(subtask_num: int,
                     query_len: int,
                     tool_num: int,
                     cross_class_num: int,
                     api_num: int,
                     experience_hit: bool,
                     traj_steps: int,
                     reuse_count: int,
                     connect_words_count: int,
                     exp_api: int) -> float:
    # ========== 1. 业务经验设定的特征极值（根据场景调整） ==========
    feature_min_max = {
        "subtask_num": (1, 5),
        "query_len": (5, 300),
        "tool_num": (1, 6),
        "cross_class_num": (1, 4),
        "api_num": (1, 15),
        "connect_words_count": (0, 10),
        "traj_steps": (1, 15),
        "exp_api": (1, 15),
        "reuse_count": (0, 50) 
    }

    # ========== 2. 特征归一化（消除量纲） ==========
    n_sub = normalize_feature(subtask_num, *feature_min_max["subtask_num"])
    n_qry = normalize_feature(query_len, *feature_min_max["query_len"])
    n_tl = normalize_feature(tool_num, *feature_min_max["tool_num"])
    n_cc = normalize_feature(cross_class_num, *feature_min_max["cross_class_num"])
    n_api = normalize_feature(api_num, *feature_min_max["api_num"])
    n_op = normalize_feature(connect_words_count, *feature_min_max["connect_words_count"])

    # ========== 3. 自定义特征权重（和为1，可根据业务调整） ==========
    weights = {
        "subtask_num": 0.35,
        "query_len": 0.05,
        "tool_num": 0.2,
        "cross_class_num": 0.25,
        "api_num": 0.05,
        "connect_words_count": 0.1
    }

    # ========== 4. 基础复杂度计算 ==========
    base_score = (
        n_sub * weights["subtask_num"] +
        n_qry * weights["query_len"] +
        n_tl * weights["tool_num"] +
        n_cc * weights["cross_class_num"] +
        n_api * weights["api_num"] +
        n_op * weights["connect_words_count"]
    )

    # ========== 5. 模块化经验修正因子 ==========
    exp_correction = 0.0
    if experience_hit:
        n_traj = normalize_feature(traj_steps, *feature_min_max["traj_steps"])
        n_exp_api = normalize_feature(exp_api, *feature_min_max["exp_api"])
        n_reuse = normalize_feature(reuse_count, *feature_min_max["reuse_count"])
        
        alpha, beta, gamma = 0.08, 0.07, 0.1
        exp_correction = alpha * n_traj + beta * n_exp_api - gamma * n_reuse
        exp_correction = max(min(exp_correction, 0.15), -0.1)

    # ========== 6. 核心分（基础分 + 修正因子） ==========
    core_score = max(base_score + exp_correction, 0.0) 

    # ========== 7. Sigmoid非线性修正（模拟复杂度饱和，输出[0,1]） ==========
    k = 10  
    b = 0.5 
    final_score = 1 / (1 + math.exp(-k * (core_score - b)))

    return final_score

def select_strategy(query: str,
                    subtasks: List[str],
                    api_pool: List[Dict],
                    experience: Optional[Experience]) -> str:
    subtask_num = len(subtasks)
    query_len = len(query.split())
    tool_names = set(t['tool_name'] for t in api_pool)
    tool_num = len(tool_names)
    category_names = set(t['category_name'] for t in api_pool)
    cross_class_num = len(category_names)
    api_num = len(api_pool)
    connect_words_count = count_connect_words(query)
    experience_hit = experience is not None
    traj_steps = 0
    reuse_count = 0
    exp_api = 0
    if experience_hit:
        traj_steps = len(experience['trajectory'].tool_order)-1 if experience.get('trajectory') else 0
        reuse_count = experience.get("reuse_count", 0)
        tool_details = experience.get("tool_details", [])
        exp_api = len(tool_details)

    score = complexity_score(subtask_num, query_len, tool_num, cross_class_num, api_num,
                             experience_hit, traj_steps, reuse_count, connect_words_count, exp_api)
    
    print("Final Complexity Score (0-1):", round(score, 4))

    threshold_low = 0.4  
    threshold_high = 0.7
    if score < threshold_low:
        return 'react'
    elif score < threshold_high:
        return 'dfs'
    else:
        return 'lats'
