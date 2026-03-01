#!/usr/bin/env python3
# l2_cache_manager.py
"""
二级缓存管理器（中等规模场景）
功能改动点：
 - 完全去掉 transformer
 - 分组匹配机制增加（按工具/API模块或子任务关键字）
 - TF-IDF + 成功率 + 最近使用时间融合评分
 - 数据结构：groups -> list of entries, entry保持与L1一致
 - 高置信匹配和候选列表模式
"""

import json
import os
import re
import time
import tempfile
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ---------- util functions ----------
def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def atomic_write(path: str, data: str):
    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass

class L2CacheManager:
    def __init__(self, cache_path: str = "./data/l2_cache.json", max_cache_size: Optional[int] = None):
        self.cache_path = cache_path
        self.entries: Dict[str, List[Dict[str, Any]]] = {}  # Store tool clusters by cluster name
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.cache_keys_norm: List[str] = []
        self.max_cache_size = max_cache_size
        self.load_cache()
        retrieval_model_path = "../../transformers/2025-07-24_02-49-05"
        self.model = SentenceTransformer(retrieval_model_path)

    def load_cache(self):
        """Load the tool clusters from the JSON cache file."""
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Error loading JSON data: {e}")
                    data = {}
        else:
            data = {}

        self.entries = data  # Directly load the clusters of tools
        self._build_tfidf_index()

    def save_cache(self):
        """Save the tool clusters back to the cache file."""
        atomic_write(self.cache_path, json.dumps(self.entries, ensure_ascii=False, indent=2))

    def _build_tfidf_index(self):
        """Build the TF-IDF index based on the tool descriptions (from 'text' field)."""
        all_descriptions = []
        # Collect all descriptions (text) from all tools in all clusters
        for cluster_name, cluster in self.entries.items():
            for tool in cluster["tools"]:  
                all_descriptions.append(tool.get("text", ""))  

        self.tfidf_vectorizer = TfidfVectorizer().fit(all_descriptions)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(all_descriptions)

    def lookup(self, query: str, label: List[str] = [], top_k: int = 3, thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Find the most relevant tool based on the task description and given label."""
        if thresholds is None:
            thresholds = {"high": 0.7, "low": 0.45}

        label = normalize_text(label)
        # First step: Find the relevant toolset based on the provided label (directly passed as input)
        relevant_toolsets = []
        for cluster_name, cluster in self.entries.items():
            if label.lower() in cluster["labels"] :  
                relevant_toolsets.append(cluster["tools"])  

        if not relevant_toolsets:
            print('"mode": "NO_HIT", "entry": None, "candidates": None, "score": 0.0')
            return None

        # # Use TF-IDF to find the most relevant tool in the selected toolset
        # candidates = []
        # for cluster in relevant_toolsets:
        #     for tool in cluster:
        #         tool_desc = tool.get("text", "")  # Use 'text' field for tool description
        #         print(tool_desc)
        #         # Calculate similarity between the tool's description and the query
        #         qvec = self.tfidf_vectorizer.transform([tool_desc])
        #         # print(qvec)
        #         sim_lex = cosine_similarity(qvec, self.tfidf_matrix)[0]
        #         top_idxs = np.argsort(sim_lex)[-top_k:][::-1]
        #         for idx in top_idxs:
        #             # Ensure idx is within the range of the cluster length
        #             if idx < len(cluster):
        #                 e = cluster[idx]
        #                 sim_e = float(sim_lex[idx])
        #                 # Only using TF-IDF similarity to calculate score
        #                 candidates.append((e, sim_e))

        # candidates.sort(key=lambda x: x[1], reverse=True)
        # best_e, best_score = candidates[0]

        # if best_score >= thresholds["high"]:
        #     return {"mode": "L2_high", "entry": best_e, "candidates": None, "score": float(best_score)}
        # elif best_score >= thresholds["low"]:
        #     top_res = [(c[0], float(c[1])) for c in candidates[:top_k]]
        #     return {"mode": "L2_candidates", "entry": None, "candidates": top_res, "score": float(best_score)}
        # else:
        #     return {"mode": "NO_HIT", "entry": None, "candidates": None, "score": float(best_score)}

        # Encode tool description using SentenceTransformer
        candidates = []
        query_embedding = self.model.encode([query])[0]
        for cluster in relevant_toolsets:
            for tool in cluster:
                tool_desc = tool.get("text", "")  
                tool_embedding = self.model.encode([tool_desc])[0]
                sim_score = cosine_similarity([query_embedding], [tool_embedding])[0][0]

                if sim_score >= thresholds["low"]:
                    candidates.append((tool, sim_score))

        if not candidates:
            print('"mode": "NO_HIT", "entry": None, "candidates": None, "score": 0.0')
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        print("candidatae:",candidates)

        if candidates:
            best_e, best_score = candidates[0]
            best_score_float = float(best_score) 
            if best_score >= thresholds["low"]:
                print("best_e:",best_e)
                info = {
                    "category_name": best_e.get("category_name"),
                    "tool_name": best_e.get("tool_name"),  
                    "api_name": best_e.get("api_name") 
                }
                print(f'"mode": "L2_high", "entry": {best_e}, "candidates": None, "score": {best_score_float}')
                return info
        else:
            print('"mode": "NO_HIT", "entry": None, "candidates": None, "score": 0.0')
            return None

    def add_entry(self, cluster_name: str, tool: str, desc: str, tags: List[str], example_queries: Optional[List[str]] = None):
        """Add a new tool entry to a specific toolset cluster."""
        if cluster_name not in self.entries:
            self.entries[cluster_name] = []
        
        entry = {
            "task_signature": {"tool": tool, "desc": desc},
            "tags": tags,
            "stats": {"usage": 0, "success": 0, "last_used": int(time.time())},
            "_key_norm": normalize_text(desc)
        }
        if example_queries:
            entry["example_queries"] = example_queries
        
        self.entries[cluster_name].append(entry)
        self._build_tfidf_index()
        self.save_cache()

    def update_on_result(self, entry: Dict[str, Any], success: bool):
        """Update the usage and success stats for the tool entry."""
        key = entry.get("_key_norm") or normalize_text(entry["task_signature"]["desc"])
        for cluster in self.entries.values():
            for e in cluster:
                if e["_key_norm"] == key:
                    e["stats"]["usage"] = e["stats"].get("usage", 0) + 1
                    if success:
                        e["stats"]["success"] = e["stats"].get("success", 0) + 1
                    e["stats"]["last_used"] = int(time.time())
                    self.save_cache()
                    return e
        return None

if __name__ == "__main__":
    # Create an instance of the cache manager
    l2_cache_manager = L2CacheManager(cache_path="l2_cache.json", max_cache_size=5)

    # Simulate a task query to search for relevant tools
    task_query = "Get Product Detail By Provide Slug"
    t=time.time()
    result = l2_cache_manager.lookup(query=task_query, label="commerce", top_k=3)
    print(time.time()-t)

    # Print out the results
    print(f"Search results for '{task_query}':")
    print(json.dumps(result, ensure_ascii=False, indent=2))
