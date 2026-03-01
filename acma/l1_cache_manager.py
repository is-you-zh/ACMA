import json
import os
import time
import tempfile
import argparse
import re
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
            try:
                os.remove(tmp)
            except Exception:
                pass

class L1CacheManager:
    def __init__(self, cache_path: str = "./data/l1_cache.json", api_lib_path: Optional[str] = None, max_cache_size: Optional[int] = 500):
        self.cache_path = cache_path
        self.entries: List[Dict[str, Any]] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.cache_keys_norm: List[str] = []
        self.api_lib = []
        self.max_cache_size = max_cache_size
        if api_lib_path:
            self._load_api_lib(api_lib_path)
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
        else:
            data = []
        self.entries = []
        for e in data:
            entry = dict(e)
            if "task_signature" not in entry:
                entry = {"task_signature": {"tool": e.get("tool")}}
            if "stats" not in entry:
                entry["stats"] = {"usage": 0, "success": 0, "last_used": 0}
            if "aliases" not in entry:
                entry["aliases"] = []
            entry["_key_norm"] = entry["_key_norm"]
            self.entries.append(entry)
        self._build_tfidf_index()
        self._enforce_cache_limit()

    def save_cache(self):
        atomic_write(self.cache_path, json.dumps(self.entries, ensure_ascii=False, indent=2))

    def _build_tfidf_index(self):
        keys = [e["_key_norm"] for e in self.entries]
        self.cache_keys_norm = keys
        t_s = time.time()
        if keys:
            self.tfidf_vectorizer = TfidfVectorizer().fit(keys)
            self.tfidf_matrix = self.tfidf_vectorizer.transform(keys)
        else:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
        print(f"the time of tfidf index building: {time.time() - t_s}")

    def _load_api_lib(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
            self.api_lib = data
        else:
            self.api_lib = []

    def _compute_succ_rate(self, entry):
        usage = entry["stats"].get("usage", 0)
        succ = entry["stats"].get("success", 0)
        return (succ + 1) / (usage + 2)

    def _enforce_cache_limit(self):
        if self.max_cache_size is not None and len(self.entries) > self.max_cache_size:
            self.entries.sort(key=lambda e: e["stats"].get("last_used", 0))
            self.entries = self.entries[-self.max_cache_size:]
            self._build_tfidf_index()
            self.save_cache()

    def lookup(self, query: str, top_k: int = 5, thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        if thresholds is None:
            thresholds = {"high": 0.75, "low": 0.45} 
        
        qnorm = normalize_text(query)
        for e in self.entries:
            if qnorm == e["_key_norm"]:
                print(f'"mode": "L1_exact", "entry": {e}, "candidates": None, "score": 1.0')
                self.update_on_result(e, True)
                return e["task_signature"]["tool"]

        for e in self.entries:
            for alias in e.get("aliases", []):
                if qnorm == normalize_text(alias):
                    print(f'"mode": "L1_alias", "entry": {e}, "candidates": None, "score": 0.99')
                    self.update_on_result(e, True)
                    return e["task_signature"]["tool"]

        if self.tfidf_vectorizer is None:
            return None

        qvec = self.tfidf_vectorizer.transform([qnorm])
        sim_lex = cosine_similarity(qvec, self.tfidf_matrix)[0]
        top_idxs = np.argsort(sim_lex)[-top_k:][::-1]

        candidates = []
        for idx in top_idxs:
            e = self.entries[int(idx)]
            sim_e = float(sim_lex[int(idx)])

            if sim_e < thresholds["high"]:
                continue
            
            succ_rate = self._compute_succ_rate(e)
            recency = np.exp(-1e-6 * max(0, time.time() - e["stats"].get("last_used", 0)))
            alpha, beta, gamma = (0.6, 0.3, 0.1)
            score = alpha * sim_e + beta * succ_rate + gamma * recency
            candidates.append((e, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_e, best_score = candidates[0]


        if best_score < thresholds["low"]:
            return None

        print(f'"mode": "L1", "entry": {best_e}, "candidates": None, "score": {float(best_score)}')
        self.update_on_result(e, True)
        return best_e["task_signature"]["tool"]
    
        # # return top_k
        # top_res = [c[0]["task_signature"]["tool"] for c in candidates[:top_k]]
        # print(f'"mode": "L1", "entry": None, "candidates": {top_res}, "score": {float(best_score)}')
        # return top_res

    def add_entry(self, tool: str, desc: str, example_queries: Optional[List[str]] = None, aliases: Optional[List[str]] = None):
        entry = {
            "task_signature": {"tool": tool},
            "stats": {"usage": 0, "success": 0, "last_used": int(time.time())},
            "aliases": aliases or [],
            "_key_norm": normalize_text(desc)
        }
        if example_queries:
            entry["example_queries"] = example_queries
        self.entries.append(entry)
        self._build_tfidf_index()
        self._enforce_cache_limit()
        self.save_cache()
        return entry

    def update_on_result(self, entry: Dict[str, Any], success: bool):
        key = entry.get("_key_norm")
        for e in self.entries:
            if e["_key_norm"] == key:
                e["stats"]["usage"] = e["stats"].get("usage", 0) + 1
                if success:
                    e["stats"]["success"] = e["stats"].get("success", 0) + 1
                e["stats"]["last_used"] = int(time.time())
                self._enforce_cache_limit()
                self.save_cache()
                return e
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default="cache.json", help="Path to cache file")
    parser.add_argument("--query", type=str, default="Retrieve the latest news headlines by category", help="Query string to lookup")
    parser.add_argument("--add", type=str, default=None, help="Add new entry, format: TOOL||DESC")
    parser.add_argument("--update-success", type=str, default=None, help="Update stats: pass normalized desc to mark last used success")
    parser.add_argument("--max-size", type=int, default=None, help="Optional max cache size for LRU eviction")
    args = parser.parse_args()

    mgr = L1CacheManager(cache_path=args.cache, api_lib_path=args.api_lib, max_cache_size=args.max_size)

    if args.add:
        if "||" not in args.add:
            print("Add format: TOOL||DESC")
            return
        tool, desc = args.add.split("||", 1)
        e = mgr.add_entry(tool.strip(), desc.strip())
        print("Added entry:", json.dumps(e, ensure_ascii=False, indent=2))
        return

    if args.update_success:
        qnorm = normalize_text(args.update_success)
        found = next((e for e in mgr.entries if e["_key_norm"] == qnorm), None)
        if found:
            mgr.update_on_result(found, True)
            print("Updated stats for:", found["task_signature"]["desc"])
        else:
            print("No entry found for that desc.")
        return

    if args.query:
        out = mgr.lookup(args.query)
        if out["mode"] in ("L1_exact", "L1_alias", "L2_high"):
            e = out["entry"]
            print("HIT MODE:", out["mode"])
            print("Tool:", e["task_signature"]["tool"])
            print("Desc:", e["task_signature"]["desc"])
            print("Score:", out["score"])
            mgr.update_on_result(e, True)
        else:
            print("No API lib provided for fallback. Consider adding --api-lib <path> or add cache entries.")


if __name__ == "__main__":
    t=time.time()
    main()
    print(time.time()-t)