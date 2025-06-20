import numpy as np
from typing import List, Tuple, Optional
import os
import pickle

class MPRMemoryStore:
    def __init__(self, embed_dim: int = 128, persist_path: Optional[str] = None):
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
