"""
🎯 minlp_pearson_retriever.py
模块功能：基于 MINLP 优化筛选结构合理的记忆候选，然后使用皮尔逊相关系数进一步判定是否召回。
用于 Agent AI 系统中的高精度、可解释记忆检索。
"""



import os
import pickle
import numpy as np
from scipy.stats import pearsonr
from typing import List, Tuple, Optional, Union

class MemoryRetriever:
    def __init__(self, memory_embeddings, memory_triples, similarity_threshold=0.7, embed_dim=384):
        """
        memory_embeddings: List[np.array] - 所有记忆三元组的向量嵌入
        memory_triples: List[Tuple[str, str, str]] - 对应的三元组内容
        """
        self.embed_dim = embed_dim
        self.threshold = similarity_threshold

        self.embeds: List[np.ndarray] = [self._resize(vec) for vec in memory_embeddings]
        self.triples: List[Tuple[str, str, str]] = memory_triples

    def _resize(self, vec: np.ndarray) -> np.ndarray:
        """确保向量为 self.embed_dim 维"""
        if vec.ndim > 1:
            vec = vec.flatten()
        if vec.shape[0] != self.embed_dim:
            vec = np.resize(vec, (self.embed_dim,))
        return vec.astype(np.float32)

    def set_memory(self, memory_embeddings: List[np.ndarray], memory_triples: List[Tuple[str, str, str]]):
        """更新记忆库内容，替换旧的记忆向量和三元组"""
        self.embeds = [self._resize(v) for v in memory_embeddings]
        self.triples = memory_triples

    def _solve_naive(self, query_vec, top_k=10):
        """简化方法：按点积排序"""
        query_vec = self._resize(query_vec)
        sims = [float(np.dot(query_vec, vec)) for vec in self.embeds]
        top_k = int(top_k)
        sorted_indices = np.argsort(sims)[::-1][:top_k]
        return sorted_indices.tolist()

    def _pearson_filter(self, query_vec, candidate_indices):
        query_vec = self._resize(query_vec)
        accepted = []
        for i in candidate_indices:
            try:
                r, _ = pearsonr(query_vec, self.embeds[i])
                if r > self.threshold:
                    accepted.append(i)
            except Exception:
                continue
        return accepted

    def retrieve(self, *, vectors=None, top_k=None, keys=None) -> List[List[int]]:
        """
        根据向量检索返回 triple 的索引列表，或根据关键词 keys 检索内容。
        只能提供 top_k 或 keys 其中一个。
        """
        if (top_k is None) == (keys is None):
            raise ValueError("必须且只能提供 top_k 或 keys 参数中的一个")

        results = []

        if vectors is not None:
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            if not isinstance(vectors, (list, tuple)):
                raise TypeError(f"vectors 必须是列表或元组，收到：{type(vectors)}")

            vectors = [self._resize(np.array(v)) for v in vectors]  # ✅ 统一为 embed_dim 维度

        if keys is not None:
            if not isinstance(keys, (list, tuple)):
                raise TypeError(f"keys 必须是列表或元组，收到：{type(keys)}")

        if top_k is not None:
            try:
                top_k = int(top_k)
            except Exception:
                raise ValueError(f"top_k必须是整数，收到：{top_k}")

            for query_vec in vectors:
                sims = []
                for idx, mem_vec in enumerate(self.embeds):
                    sim = np.dot(query_vec, mem_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(mem_vec) + 1e-10)
                    sims.append((idx, sim))

                sims.sort(key=lambda x: x[1], reverse=True)
                filtered = [idx for idx, sim in sims if sim >= self.threshold][:top_k]
                results.append(filtered)

        elif keys is not None:
            for key in keys:
                matched = [i for i, triple in enumerate(self.triples) if key in triple]
                results.append(matched if matched else [])

        return results






class MPRMemoryStore:
    def __init__(self, embed_dim: int = 384, persist_path: Optional[str] = None):
        self.embed_dim = embed_dim
        self.memory_keys: List[Tuple[str, str, str]] = []  # 三元组
        self.memory_vectors: List[np.ndarray] = []         # 向量
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self.load_memory()

    def set_memory(self, vectors: List[np.ndarray], keys: List[Tuple[str, str, str]]):
        """
        用于更新记忆库内容（全量替换）
        """
        if len(vectors) != len(keys):
            raise ValueError("向量和三元组数量不一致")

        self.memory_keys = keys
        self.memory_vectors = [self._normalize_vector(v) for v in vectors]

        if self.persist_path:
            self.save_memory()

    def retrieve(self, query_vector: np.ndarray, top_k: int = 5) -> List[int]:
        """
        返回最相似的 top_k 记忆向量索引（基于余弦相似度）
        """
        if not self.memory_vectors:
            return []

        query_vec = self._normalize_vector(query_vector)
        sims = [self._cosine_similarity(query_vec, v) for v in self.memory_vectors]
        sorted_indices = np.argsort(sims)[::-1]  # 从大到小
        return sorted_indices[:top_k].tolist()

    def get_triple_by_index(self, idx: int) -> Tuple[str, str, str]:
        return self.memory_keys[idx]

    def save_memory(self):
        if not self.persist_path:
            return
        with open(self.persist_path, "wb") as f:
            pickle.dump({
                "keys": self.memory_keys,
                "vectors": self.memory_vectors
            }, f)

    def load_memory(self):
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
            self.memory_keys = data["keys"]
            self.memory_vectors = data["vectors"]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        if vec.ndim > 1:
            vec = vec.flatten()
        if vec.shape[0] != self.embed_dim:
            vec = np.resize(vec, (self.embed_dim,))
        return vec.astype(np.float32)
