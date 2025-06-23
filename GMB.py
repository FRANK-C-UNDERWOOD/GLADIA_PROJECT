import json
import torch
from typing import Tuple, List, Dict
import os # Added for os.path.exists

class GraphMemoryBank:
    def __init__(self):
        self.graph_nodes: Dict[str, Dict] = {}  # 节点信息（实体/关系）
        self.graph_edges: List[Dict] = []       # 三元组边

    def add_triplet(self, triple: Tuple[str, str, str], vector: torch.Tensor, error: float, activation: float = 1.0):
        s, p, o = triple
        s_id = s
        o_id = o

        # Ensure vector is a 1D tensor before converting to list
        if vector.ndim > 1:
            squeezed_vector = vector.squeeze().tolist()
        else:
            squeezed_vector = vector.tolist()

        # 添加节点
        for node_id in [s_id, o_id]:
            if node_id not in self.graph_nodes:
                self.graph_nodes[node_id] = {
                    "type": "entity",
                    "coords": torch.rand(2).tolist(), # Example coordinates
                    "vector": squeezed_vector, # Store as list
                    "activation": activation
                }

        if p not in self.graph_nodes:
            self.graph_nodes[p] = {
                "type": "relation" # Relations might not have vectors or detailed data
            }

        # 添加边（允许重复记录）
        self.graph_edges.append({
            "from": s_id,
            "to": o_id,
            "label": p,
            "vector": squeezed_vector, # Store triplet vector as list with the edge
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
        # graph_nodes and graph_edges store vectors as lists (from add_triplet).
        # These are directly JSON serializable.
        graph_data_to_save = {
            "nodes": self.graph_nodes,
            "edges": self.graph_edges
        }
        try:
            with open(f"{path_prefix}.graph.json", "w", encoding="utf-8") as f:
                json.dump(graph_data_to_save, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[❌] GMB Save: Error saving JSON file {path_prefix}.graph.json: {e}")
            # Depending on desired robustness, may want to return or raise here.

        # For .pt file, convert lists from self.graph_nodes and self.graph_edges to tensors.
        node_tensors_for_pt = {}
        if self.graph_nodes:  # Check if graph_nodes is not None and not empty
            for node_id, node_data in self.graph_nodes.items():
                if node_data and node_data.get("type") == "entity" and "vector" in node_data:
                    vector_list = node_data["vector"]
                    if isinstance(vector_list, list):
                        try:
                            node_tensors_for_pt[node_id] = torch.tensor(vector_list, dtype=torch.float32)
                        except Exception as e:
                            print(f"[⚠️] GMB Save: Error converting node vector to tensor for node '{node_id}'. Skipping. Error: {e}")
                    elif isinstance(vector_list, torch.Tensor): # Should ideally be list from add_triplet
                        node_tensors_for_pt[node_id] = vector_list.float() 
                    # Else: vector is not a list or tensor, skip.

        edge_tensors_for_pt = []
        if self.graph_edges:  # Check if graph_edges is not None and not empty
            for idx, edge_data in enumerate(self.graph_edges):
                if edge_data and "vector" in edge_data:
                    vector_list = edge_data["vector"]
                    if isinstance(vector_list, list) and vector_list:  # Ensure list is not empty
                        try:
                            edge_tensors_for_pt.append(torch.tensor(vector_list, dtype=torch.float32))
                        except Exception as e:
                            print(f"[⚠️] GMB Save: Error converting edge vector to tensor for edge index {idx}. Appending empty tensor. Error: {e}")
                            edge_tensors_for_pt.append(torch.empty(0, dtype=torch.float32))
                    elif isinstance(vector_list, torch.Tensor): # Should ideally be list
                        edge_tensors_for_pt.append(vector_list.float())
                    elif not vector_list: # Handles empty list from "vector": []
                        edge_tensors_for_pt.append(torch.empty(0, dtype=torch.float32))
                    # Else: vector is not a list, not a tensor, or an unhandled type, append empty.
                    else:
                        print(f"[⚠️] GMB Save: Edge vector for edge index {idx} is of unexpected type or empty. Appending empty tensor.")
                        edge_tensors_for_pt.append(torch.empty(0, dtype=torch.float32))
                else: # edge_data is None or "vector" key is missing
                    edge_tensors_for_pt.append(torch.empty(0, dtype=torch.float32))


        data_to_save_pt = {
            "node_vectors": node_tensors_for_pt,
            "edge_vectors": edge_tensors_for_pt
        }
        try:
            torch.save(data_to_save_pt, f"{path_prefix}.pt")
            print(f"[💾] GMB: 已保存图结构记忆至 {path_prefix}.graph.json / .pt (Nodes: {len(self.graph_nodes)}, Edges: {len(self.graph_edges)})")
        except Exception as e:
            print(f"[❌] GMB Save: Failed to save .pt file {path_prefix}.pt: {e}")

    def load_all(self, path_prefix: str = "dialog_memory"):
        # Initialize to empty in case files are missing or corrupted
        self.graph_nodes = {}
        self.graph_edges = []

        json_file_path = f"{path_prefix}.graph.json"
        pt_file_path = f"{path_prefix}.pt"

        # Load from JSON
        if not os.path.exists(json_file_path):
            print(f"[⚠️] GMB Load: JSON file not found: {json_file_path}. Creating an empty one.")
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump({"nodes": {}, "edges": []}, f, ensure_ascii=False, indent=2)
        else:
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    graph_data_loaded = json.load(f)
                    self.graph_nodes = graph_data_loaded.get("nodes", {})
                    self.graph_edges = graph_data_loaded.get("edges", [])
            except json.JSONDecodeError:
                print(f"[⚠️] GMB Load: Error decoding JSON from {json_file_path}. File might be corrupted.")
            except Exception as e:
                print(f"[❌] GMB Load: Generic error loading JSON file {json_file_path}: {e}.")

        # Load from .pt and potentially override/initialize vectors
        node_tensors_from_pt = {}
        edge_tensors_from_pt = []

        if not os.path.exists(pt_file_path):
            print(f"[⚠️] GMB Load: PyTorch file not found: {pt_file_path}. Creating an empty one.")
            try:
                torch.save({"node_vectors": {}, "edge_vectors": []}, pt_file_path)
            except Exception as e_save:
                print(f"[❌] GMB Load: Failed to create empty {pt_file_path}: {e_save}")
        else:
            try:
                # Add map_location for robustness if file was saved on a different device (e.g. GPU vs CPU)
                pt_data = torch.load(pt_file_path, map_location=torch.device('cpu'))
                node_tensors_from_pt = pt_data.get("node_vectors", {})
                edge_tensors_from_pt = pt_data.get("edge_vectors", [])
            except Exception as e: 
                print(f"[⚠️] GMB Load: Error loading PyTorch file {pt_file_path}: {e}. Trying to re-initialize.")
                try: 
                    torch.save({"node_vectors": {}, "edge_vectors": []}, pt_file_path)
                except Exception as e_save_fresh:
                    print(f"[❌] GMB Load: Failed to re-initialize corrupted {pt_file_path}: {e_save_fresh}")

        # Integrate vectors: .pt tensors take precedence. If not, convert JSON lists to tensors.
        for node_id, node_data in self.graph_nodes.items():
            if node_data and node_data.get("type") == "entity":
                # Check PT file first
                if node_id in node_tensors_from_pt and isinstance(node_tensors_from_pt[node_id], torch.Tensor):
                    node_data["vector"] = node_tensors_from_pt[node_id]
                # Else, check JSON list and convert
                elif "vector" in node_data and isinstance(node_data["vector"], list):
                    try:
                        node_data["vector"] = torch.tensor(node_data["vector"], dtype=torch.float32)
                    except Exception as e:
                        print(f"[⚠️] GMB Load: Error converting node vector list for '{node_id}' to tensor. Removing. Error: {e}")
                        node_data.pop("vector", None)
                # Else if vector exists but isn't list or already tensor (e.g. from older format), try to remove
                elif "vector" in node_data:
                     print(f"[⚠️] GMB Load: Node '{node_id}' vector has unexpected format. Removing vector.")
                     node_data.pop("vector", None)
        
        processed_edges = []
        if self.graph_edges:
            # If PT edge vectors align with JSON edges in count
            if edge_tensors_from_pt and len(self.graph_edges) == len(edge_tensors_from_pt):
                for i, edge_data in enumerate(self.graph_edges):
                    if edge_data: # Ensure edge_data is not None
                        current_edge_tensor = edge_tensors_from_pt[i]
                        if isinstance(current_edge_tensor, torch.Tensor) and current_edge_tensor.numel() > 0:
                            edge_data["vector"] = current_edge_tensor
                        # If PT tensor is empty, but JSON list has data, prefer JSON list converted to tensor
                        elif "vector" in edge_data and isinstance(edge_data["vector"], list) and edge_data["vector"]:
                            try:
                                edge_data["vector"] = torch.tensor(edge_data["vector"], dtype=torch.float32)
                            except Exception as e:
                                print(f"[⚠️] GMB Load (PT-align path): Error converting edge vector list for edge index {i} to tensor. Removing. Error: {e}")
                                edge_data.pop("vector", None)
                        else: # PT tensor empty, JSON vector also not usable or missing
                            edge_data.pop("vector", None)
                        processed_edges.append(edge_data)
            # Fallback: PT edge vectors missing or mismatched count, rely on JSON lists
            else:
                if edge_tensors_from_pt: # Log if there was a mismatch
                     print(f"[⚠️] GMB Load: Mismatch between JSON edges ({len(self.graph_edges)}) and PT edge_vectors ({len(edge_tensors_from_pt)}). Using JSON lists for vectors.")
                for i, edge_data in enumerate(self.graph_edges):
                    if edge_data: # Ensure edge_data is not None
                        if "vector" in edge_data and isinstance(edge_data["vector"], list) and edge_data["vector"]:
                            try:
                                edge_data["vector"] = torch.tensor(edge_data["vector"], dtype=torch.float32)
                            except Exception as e:
                                print(f"[⚠️] GMB Load (JSON fallback): Error converting edge vector list for edge index {i} to tensor. Removing. Error: {e}")
                                edge_data.pop("vector", None)
                        # If vector from JSON is not a valid list or missing
                        else:
                            edge_data.pop("vector", None)
                        processed_edges.append(edge_data)
        self.graph_edges = processed_edges

        # Final cleanup pass for any vectors that are not tensors
        for node_data in self.graph_nodes.values():
            if node_data and "vector" in node_data and not isinstance(node_data.get("vector"), torch.Tensor):
                node_id_log = next(iter(self.graph_nodes.keys())) # get some id for logging
                print(f"[🔧] GMB Load (Final Check): Node vector for '{node_id_log}' was not a Tensor. Removing.")
                node_data.pop("vector", None)

        for edge_data in self.graph_edges:
            if edge_data and "vector" in edge_data and not isinstance(edge_data.get("vector"), torch.Tensor):
                edge_label_log = edge_data.get('label', 'UNKNOWN_EDGE')
                print(f"[🔧] GMB Load (Final Check): Edge vector for '{edge_label_log}' was not a Tensor. Removing.")
                edge_data.pop("vector", None)
        
        print(f"[✅] GMB: 已完成加载图记忆: {path_prefix} (Nodes: {len(self.graph_nodes)}, Edges: {len(self.graph_edges)})")


    def stats(self) -> Dict[str, int]:
        return {
            "node_count": len(self.graph_nodes),
            "edge_count": len(self.graph_edges),
            "entity_count": sum(1 for v in self.graph_nodes.values() if v.get("type") == "entity"),
            "relation_count": sum(1 for v in self.graph_nodes.values() if v.get("type") == "relation")
        }

    def recall_memory(self, triple: Tuple[str, str, str]) -> List[Tuple[str, str, str]]:
        s, p, o = triple
        matches = []
        for edge in self.graph_edges:
            if edge: # Ensure edge is not None
                if s in edge.get("from", "") or p in edge.get("label", "") or o in edge.get("to", ""):
                    matches.append((edge.get("from"), edge.get("label"), edge.get("to")))
        return matches

    def decay_activations(self, decay_factor=0.95):
        for node_data in self.graph_nodes.values():
            if node_data and "activation" in node_data: # Ensure node_data is not None
                node_data["activation"] *= decay_factor

