import json
import torch
from typing import Tuple, List, Dict


class GraphMemoryBank:
    def __init__(self):
        self.graph_nodes: Dict[str, Dict] = {}  # 节点信息（实体/关系）
        self.graph_edges: List[Dict] = []       # 三元组边

    def add_triplet(self, triple: Tuple[str, str, str], vector: torch.Tensor, error: float, activation: float = 1.0):
        s, p, o = triple
        s_id = s
        o_id = o

        # 添加节点
        for node_id in [s_id, o_id]:
            if node_id not in self.graph_nodes:
                self.graph_nodes[node_id] = {
                    "type": "entity",
                    "coords": torch.rand(2).tolist(),
                    "vector": vector.squeeze(0).tolist(),
                    "activation": activation
                }

        if p not in self.graph_nodes:
            self.graph_nodes[p] = {
                "type": "relation"
            }

        # 添加边（允许重复记录）
        self.graph_edges.append({
            "from": s_id,
            "to": o_id,
            "label": p,
            "error": error,
            "activation": activation
        })

    def get_neighbors(self, node_id: str, max_hops: int = 1) -> List[str]:
        neighbors = set()
        current_level = {node_id}
        for _ in range(max_hops):
            next_level = set()
            for edge in self.graph_edges:
                if edge["from"] in current_level:
                    next_level.add(edge["to"])
                if edge["to"] in current_level:
                    next_level.add(edge["from"])
            neighbors.update(next_level)
            current_level = next_level
        return list(neighbors)

    def save_all(self, path_prefix: str = "dialog_memory"):
        graph_data = {
            "nodes": self.graph_nodes,
            "edges": self.graph_edges
        }
        with open(f"{path_prefix}.graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        torch.save({
            "node_vectors": {
                k: v["vector"]
                for k, v in self.graph_nodes.items()
                if v["type"] == "entity"
            }
        }, f"{path_prefix}.pt")

        print(f"[💾] 已保存图结构记忆至 {path_prefix}.graph.json / .pt")

    def load_all(self, path_prefix: str = "dialog_memory"):
        with open(f"{path_prefix}.graph.json", "r", encoding="utf-8") as f:
            graph_data = json.load(f)
            self.graph_nodes = graph_data.get("nodes", {})
            self.graph_edges = graph_data.get("edges", [])

        pt_data = torch.load(f"{path_prefix}.pt")
        for k, vec in pt_data.get("node_vectors", {}).items():
            if k in self.graph_nodes and self.graph_nodes[k]["type"] == "entity":
                self.graph_nodes[k]["vector"] = vec

        print(f"[✅] 已加载图记忆文件：{path_prefix}")

    def stats(self) -> Dict[str, int]:
        return {
            "node_count": len(self.graph_nodes),
            "edge_count": len(self.graph_edges),
            "entity_count": sum(1 for v in self.graph_nodes.values() if v["type"] == "entity"),
            "relation_count": sum(1 for v in self.graph_nodes.values() if v["type"] == "relation")
        }

    def recall_memory(self, triple: Tuple[str, str, str]) -> List[Tuple[str, str, str]]:
        """
        通过模糊匹配返回相似三元组（简单包含关系）
        """
        s, p, o = triple
        matches = []
        for edge in self.graph_edges:
            if s in edge["from"] or p in edge["label"] or o in edge["to"]:
                matches.append((edge["from"], edge["label"], edge["to"]))
        return matches

    def decay_activations(self, decay_factor=0.95):
        """衰减所有实体节点的 activation 值"""
        for node in self.graph_nodes.values():
            if "activation" in node:
                node["activation"] *= decay_factor
