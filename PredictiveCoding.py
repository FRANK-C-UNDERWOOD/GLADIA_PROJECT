"""
🧠 PredictiveCodingAgent 模块
Author: DOCTOR + 歌蕾蒂娅 (2025)

本模块实现一个具备感知预测、误差反向修正与记忆机制的基础预测编码 Agent。
该结构可用于主动感知、异常检测、新奇性记忆、自适应控制等场景。

核心功能：
1. encode_input()   - 将输入编码为隐状态向量
2. decode_prediction() - 从隐状态生成预测
3. forward_predict()   - 执行多轮预测-误差-修正闭环推理
4. update_memory()     - 将高预测误差的输入记入记忆库
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Union
from collections import deque
import math
from MPR import MemoryRetriever
from GMB import GraphMemoryBank
# ======================== TN-MPS 记忆编码器 ========================
class TripleCompressor:
    """🔧 TN压缩器 - 将文本三元组压缩为张量表示"""
    def __init__(self, embed_dim=32):
        self.embed_dim = embed_dim
    
    def text_to_tensor(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            text = str(text)

        # Step 1: 编码为字节
        byte_data = text.encode('utf-8', errors='replace')
        safe_length = min(len(byte_data), self.embed_dim)
        truncated = byte_data[:safe_length]
        vec = np.frombuffer(truncated, dtype=np.uint8).astype(np.float32)

        # Step 2: Padding 到指定长度
        if len(vec) < self.embed_dim:
            vec = np.pad(vec, (0, self.embed_dim - len(vec)), constant_values=0)

        # Step 3: Normalize 到 0~1
        vec /= 255.0

        # Step 4: 若全为 0，则 fallback 为 hash embedding
        if np.allclose(vec, 0.0, atol=1e-6):
            hash_bytes = hashlib.sha256(text.encode()).digest()
            hash_vec = np.frombuffer(hash_bytes[:self.embed_dim], dtype=np.uint8).astype(np.float32) / 255.0
            return hash_vec

        return vec

    
    def compress_triplet(self, triple: Tuple[str, str, str]) -> np.ndarray:
        s_vec = self.text_to_tensor(triple[0])
        p_vec = self.text_to_tensor(triple[1])
        o_vec = self.text_to_tensor(triple[2])
        tensor = np.tensordot(s_vec, p_vec, axes=0)
        tensor = np.tensordot(tensor, o_vec, axes=0)
        return tensor

class MPSMemoryEncoder(nn.Module):
    """🧠 MPS编码器 - 使用矩阵乘积状态编码序列"""
    def __init__(self, input_dim, feature_dim, bond_dim, output_dim):
        super().__init__()
        self.n_sites = input_dim
        self.feature_dim = feature_dim
        self.bond_dim = bond_dim
        self.mps_cores = nn.ParameterList([
            nn.Parameter(torch.randn(bond_dim, feature_dim, bond_dim)) for _ in range(input_dim)
        ])
        self.start = nn.Parameter(torch.randn(1, bond_dim))
        self.end = nn.Parameter(torch.randn(bond_dim, 1))
        self.fc = nn.Linear(1, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        start = self.start.expand(B, -1).unsqueeze(1)
        result = torch.matmul(start, self._contract(x[:, 0, :], self.mps_cores[0]))
        for i in range(1, self.n_sites):
            result = torch.matmul(result, self._contract(x[:, i, :], self.mps_cores[i]))
        end = self.end.unsqueeze(0).expand(B, -1, -1)
        result = torch.matmul(result, end).view(B, 1)
        result = self.fc(result)
        return self.norm(result)
    
    def _contract(self, feature_vec, mps_tensor):
        return torch.einsum('bf,dfr->bdr', feature_vec, mps_tensor)

class IntegratedTN_MPS(nn.Module):
    """🔧 TN-MPS集成系统 - 将三元组编码为记忆向量"""
    def __init__(self, 
                 tn_embed_dim=32,
                 mps_input_dim=3,
                 mps_bond_dim=16,
                 mps_output_dim=64):
        super().__init__()
        self.tn_compressor = TripleCompressor(embed_dim=tn_embed_dim)
        self.adapter = nn.Sequential(
            nn.Linear(tn_embed_dim**3, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, tn_embed_dim * 3),
            nn.ReLU()
        )
        self.mps_encoder = MPSMemoryEncoder(
            input_dim=mps_input_dim,
            feature_dim=tn_embed_dim,
            bond_dim=mps_bond_dim,
            output_dim=mps_output_dim
        )
        self.tn_embed_dim = tn_embed_dim
        self.mps_input_dim = mps_input_dim
    
    def forward(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
        batch_tensors = []
        for triple in triples:
            tn_tensor = self.tn_compressor.compress_triplet(triple)
            flat_tensor = tn_tensor.flatten()
            adapted = self.adapter(torch.tensor(flat_tensor).float())
            batch_tensors.append(adapted)
        
        batch_size = len(triples)
        seq_input = torch.stack(batch_tensors).view(batch_size, self.mps_input_dim, self.tn_embed_dim)
        return self.mps_encoder(seq_input)

# ======================== 空间增强RNN记忆库 ========================
class SpatialMemoryBank(nn.Module):
    """🧠 空间增强记忆库 - 整合seRNN的先进记忆结构"""
    def __init__(self, input_dim, hidden_dim, capacity=100, 
                 retrieval_k=5, decay_factor=0.95):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.capacity = capacity
        self.retrieval_k = retrieval_k
        self.decay_factor = decay_factor
        
        # 空间坐标系统
        self.coords = nn.Parameter(torch.randn(capacity, 2), requires_grad=False)
        
        # 记忆存储
        self.memory_keys = deque(maxlen=capacity)  # 记忆键（三元组）
        self.memory_vectors = deque(maxlen=capacity)  # 记忆向量
        self.memory_errors = deque(maxlen=capacity)  # 预测误差
        self.memory_activations = deque(maxlen=capacity)  # 空间激活值
        
        # seRNN组件
        self.input2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_weights = nn.Parameter(torch.randn(hidden_dim, 2))
        
        # 自适应检索门
        self.retrieval_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def spatial_activation(self):
        """计算所有记忆的空间激活值 - 基于几何距离"""
        if len(self.memory_vectors) == 0:
            return torch.tensor([])
        
        # 计算全局中心点
        center = self.spatial_weights.mean(dim=0)
        
        # 计算所有坐标到中心的距离
        dists = torch.norm(self.coords[:len(self.memory_vectors)] - center, dim=1)
        activations = 1.0 / (1.0 + dists)
        return activations
    
    def add_memory(self, key, vector, error):
        """添加新记忆到记忆库"""
        # 如果记忆库满了，找到激活值最低的记忆替换
        if len(self.memory_vectors) >= self.capacity:
            min_act_idx = torch.argmin(torch.tensor(self.memory_activations)).item()
            # 替换激活值最低的记忆
            self.memory_keys[min_act_idx] = key
            self.memory_vectors[min_act_idx] = vector
            self.memory_errors[min_act_idx] = error
        else:
            # 添加新记忆
            self.memory_keys.append(key)
            self.memory_vectors.append(vector)
            self.memory_errors.append(error)
        
        # 更新空间激活值
        self._update_activations()
    
    def _update_activations(self):
        """更新所有记忆的空间激活值"""
        activations = self.spatial_activation()
        if len(activations) == 0:
            self.memory_activations = deque(maxlen=self.capacity)
            return
        
        # 更新激活值队列
        self.memory_activations = deque(activations.tolist(), maxlen=self.capacity)
    
    # 在 SpatialMemoryBank 类中修改 retrieve_memories 方法
    def retrieve_memories(self, query_vector: torch.Tensor) -> Tuple[List, List[torch.Tensor]]:
        """健壮的内存检索实现 - 同时解决维度不匹配和列表索引问题"""
        if len(self.memory_vectors) == 0:
            return [], []
        
        # 步骤1: 确保查询向量维度正确 (1, D)
        query_vector = self._ensure_2d(query_vector)
        
        # 步骤2: 准备内存向量张量 (M, D)
        mem_vectors = self._prepare_memory_tensor()
        
        # 步骤3: 计算相似度 (M,)
        sim = F.cosine_similarity(query_vector.expand_as(mem_vectors), mem_vectors, dim=1)
        
        # 步骤4: 添加空间激活值
        if len(self.memory_activations) == len(sim):
            total_sim = sim + torch.tensor(self.memory_activations, device=query_vector.device)
        else:
            total_sim = sim
        
        # 步骤5: 获取Top-K索引 (健壮处理)
        k = min(self.retrieval_k, len(total_sim))
        topk_idxs = self._get_topk_indices(total_sim, k)
        
        # 步骤6: 检索结果 (统一维度)
        retrieved_keys = [self.memory_keys[i] for i in topk_idxs]
        retrieved_vectors = [self._ensure_2d(self.memory_vectors[i]) for i in topk_idxs]
        
        return retrieved_keys, retrieved_vectors
    
    def contextualize(self, query_vector: torch.Tensor, retrieved_vectors: List[torch.Tensor]) -> torch.Tensor:
        """上下文融合方法 - 处理各种向量维度"""
        if not retrieved_vectors:
            return self._ensure_2d(query_vector)
        
        # 准备输入序列：当前查询 + 检索的记忆
        vectors = [self._ensure_2d(vec) for vec in [query_vector] + retrieved_vectors]
        inputs = torch.cat(vectors, dim=0)  # (1+k, D)
        
        # seRNN处理
        h = torch.zeros(1, self.hidden_dim, device=query_vector.device)
        for t in range(inputs.size(0)):
            x_t = inputs[t].unsqueeze(0)  # (1, D)
            h = torch.tanh(self.input2hidden(x_t) + self.hidden2hidden(h))
        
        # 自适应门控融合
        gate = self.retrieval_gate(query_vector)
        contextualized = gate * h + (1 - gate) * self.input2hidden(self._ensure_2d(query_vector))
        
        return contextualized
    
    # ======== 辅助方法 ========
    
    def _ensure_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        """确保张量是二维的: (1, D)"""
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        if tensor.dim() == 3:
            return tensor.squeeze(0)
        return tensor
    
    def _prepare_memory_tensor(self) -> torch.Tensor:
        """准备内存张量 (M, D)"""
        mem_tensors = []
        for vec in self.memory_vectors:
            # 统一为2D张量并压缩批次维度
            vec_2d = self._ensure_2d(vec)
            mem_tensors.append(vec_2d.squeeze(0))  # 从(1,D)变为(D)
        
        return torch.stack(mem_tensors)  # (M, D)
    
    def _get_topk_indices(self, similarities: torch.Tensor, k: int) -> List[int]:
        """健壮获取Top-K索引，返回整数列表"""
        topk_values, topk_indices = torch.topk(similarities, k)
        
        if k == 1:
            # 单元素情况
            return [topk_indices.item()]
        
        # 多元素情况
        return topk_indices.tolist()
    
    def add_memory(self, key, vector: torch.Tensor, error):
            target_dim = 384  # 根据模型配置设定
            vector = self._ensure_2d(vector)
            if vector.size(1) != target_dim:
                if vector.size(1) < target_dim:
                   # 填充零值至目标维度
                   pad_size = target_dim - vector.size(1)
                   vector = torch.cat([vector, torch.zeros(1, pad_size)], dim=1)
                else:
                    # 裁剪至目标维度
                    vector = vector[:, :target_dim]
    
             # 2. 校验维度一致性（防御性编程）
            assert vector.size(1) == target_dim, \
                f"向量维度必须为{target_dim}，当前为{vector.size(1)}"
    
            # 3. 改进记忆替换逻辑
            if len(self.memory_vectors) >= self.capacity:
                min_act_idx = torch.argmin(torch.tensor(self.memory_activations)).item()
                # 仅当新向量维度与旧向量一致时才替换
                if self.memory_vectors[min_act_idx].size(1) == target_dim:
                    self.memory_keys[min_act_idx] = key
                    self.memory_vectors[min_act_idx] = vector
                    self.memory_errors[min_act_idx] = error
                else:
                    # 维度不匹配时跳过替换或扩容
                    self.memory_keys.append(key)
                    self.memory_vectors.append(vector)
                    self.memory_errors.append(error)
            else:
                    self.memory_keys.append(key)
                    self.memory_vectors.append(vector)
                    self.memory_errors.append(error)
    
            # 4. 更新激活状态
            self._update_activations()

    
    def contextualize(self, query_vector, retrieved_vectors):
        if not retrieved_vectors:
            return self._ensure_2d(query_vector)
        inputs = [self._ensure_2d(query_vector)] + [self._ensure_2d(vec) for vec in retrieved_vectors]
        inputs = torch.cat(inputs, dim=0)
        h = torch.zeros(1, self.hidden_dim, device=query_vector.device)
        for t in range(inputs.size(0)):
            x_t = inputs[t].unsqueeze(0)
            h = torch.tanh(self.input2hidden(x_t) + self.hidden2hidden(h))
        gate = self.retrieval_gate(query_vector)
        projected_query = self.input2hidden(self._ensure_2d(query_vector))
        contextualized = gate * h + (1 - gate) * projected_query
        return contextualized

    
    def decay_activations(self):
        """衰减记忆激活值 - 模拟记忆衰减"""
        if len(self.memory_activations) == 0:
            return
        
        # 衰减激活值
        new_activations = [act * self.decay_factor for act in self.memory_activations]
        self.memory_activations = deque(new_activations, maxlen=self.capacity)

    def get_all_memories(self):
        """获取当前所有记忆键和向量"""
        if not self.memory_keys or not self.memory_vectors:
            return [], []
        
        # 确保向量维度一致 (1, D)
        vectors = [self._ensure_2d(vec) for vec in self.memory_vectors]
        return list(self.memory_keys), vectors
# ======================== 预测编码智能体 ========================
class PredictiveCodingAgent(nn.Module):
    """🧠 预测编码智能体 - 整合TN-MPS编码器和空间记忆库"""
    def __init__(self, tn_embed_dim=32,mps_bond_dim=16,mps_output_dim=384,hidden_dim=384,memory_capacity=200,memory_threshold=0.1,graph_memory_bank=None):
        super().__init__()
        

        # 记忆编码系统
        self.memory_encoder = IntegratedTN_MPS(
            tn_embed_dim=tn_embed_dim,
            mps_bond_dim=mps_bond_dim,
            mps_output_dim=mps_output_dim
        )
        
        # 空间增强记忆库
        self.memory_bank = SpatialMemoryBank(
            input_dim=mps_output_dim,
            hidden_dim=hidden_dim,
            capacity=memory_capacity
        )
        
        # 预测编码组件
        self.input_dim = mps_output_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, self.input_dim),
            nn.LayerNorm(self.input_dim)
        )
        
        # 上下文到记忆维度的投影层
        self.context_to_memory_dim = nn.Linear(hidden_dim, mps_output_dim)
        
        # 记忆参数
        self.memory_threshold = memory_threshold
        self.memory_update_freq = 5
        self.step_count = 0

        self.MPR = MemoryRetriever(memory_embeddings=[],memory_triples=[],similarity_threshold=0.7)

        # 新增图记忆库
        self.graph_memory = graph_memory_bank or GraphMemoryBank()
    def encode_input(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
        """编码三元组为记忆向量（带梯度保护）"""
        with torch.no_grad():
            return self.memory_encoder(triples)
    
    def predict_with_memory(self, query_vector: torch.Tensor) -> torch.Tensor:
        """使用MPR检索器的记忆增强预测"""
        # 确保查询向量是二维的 (1, D)
        query_vector = self.memory_bank._ensure_2d(query_vector)
    
        retrieved_vectors = []
    
        if len(self.memory_bank.memory_keys) > 0:
            # 1. 获取当前所有记忆
            keys, vectors = self.memory_bank.get_all_memories()
        
            # 2. 准备numpy格式的记忆向量
            vectors_np = [vec.detach().cpu().numpy().flatten() for vec in vectors]
        
            # 3. 更新MPR记忆库（假设有对应方法）
            # 这里不要调用 retrieve，而是调用更新记忆库的方法，比如：
            self.MPR.set_memory(vectors_np, keys)  # 你需要自己实现这个方法
        
           # 4. 准备查询向量（numpy格式）
            query_np = query_vector.detach().cpu().numpy().flatten()
        
            # 5. 执行MPR检索，使用关键字参数
            indices = self.MPR.retrieve(vectors=[query_np], top_k=5)  # 这里传入列表包裹query_np
        
            # 6. 根据indices取对应向量（你可能需要实现retrieve返回的具体内容）
            for idx in indices[0]:  # indices[0]是第一个查询的top_k结果
                vec_np = vectors_np[idx]
                vec_tensor = torch.tensor(vec_np, device=query_vector.device, dtype=torch.float32).view(1, -1)
                retrieved_vectors.append(vec_tensor)
    
        # 7. 使用检索到的记忆进行上下文融合
        if retrieved_vectors:
            contextualized = self.memory_bank.contextualize(query_vector, retrieved_vectors)
        else:
            contextualized = query_vector
    
         # 8. 投影到目标维度
        if contextualized.size(-1) == self.hidden_dim:
            contextualized = self.context_to_memory_dim(contextualized)
    
        # 9. 预测
        h = self.encoder(contextualized)
        return self.decoder(h)



    def forward(self, triples: List[Tuple[str, str, str]]) -> Tuple[torch.Tensor, float]:
        # 1. 编码输入为记忆向量
        memory_vector = self.encode_input(triples)
        
        # 2. 循环处理每个三元组
        predictions = []
        errors = []
        for i in range(len(triples)):
            # 处理单个三元组
            single_vector = memory_vector[i].unsqueeze(0)
            triple = triples[i]
            
            # 3. 记忆增强预测
            prediction = self.predict_with_memory(single_vector)
            predictions.append(prediction)
            
            # 4. 计算预测误差
            with torch.no_grad():
                error = F.mse_loss(prediction, single_vector).item()
                errors.append(error)
            
            # 5. 更新记忆库（如果误差高）
            if error > self.memory_threshold:
                self.memory_bank.add_memory(
                    key=triple,
                    vector=single_vector.detach().clone(),
                    error=error
                )
                
                # 新增：添加到图记忆库
                self.graph_memory.add_triplet(
                    triple=triple,
                    vector=single_vector.detach().clone(),
                    error=error
                )
        
        # 6. 定期维护记忆库
        self.step_count += 1
        if self.step_count % self.memory_update_freq == 0:
            self.memory_bank.decay_activations()
        
        predictions_tensor = torch.cat(predictions, dim=0)
        avg_error = sum(errors) / len(errors) if errors else 0.0
        return predictions_tensor, avg_error
    
    def recall_memory(self, query_triple: Tuple[str, str, str]) -> List[Tuple[str, str, str]]:
        # 1. 从图记忆库中检索
        graph_results = []
        for ent in [query_triple[0], query_triple[2]]:
            graph_results.extend(self.graph_memory.get_neighbors(ent))
    
        """使用MPR检索器回忆与查询相关的记忆"""
        # 1. 编码查询三元组
        query_vector = self.encode_input([query_triple])
        query_vector = self.memory_bank._ensure_2d(query_vector[0])

        # 2. 准备查询向量（numpy格式）
        query_np = query_vector.detach().cpu().numpy().flatten()

        # 3. 执行检索：MPR 内部负责向量比对 + 返回三元组
        # 这里改为用关键字参数 vectors，且传入列表形式
        retrieved_triples = self.MPR.retrieve(vectors=[query_np.tolist()], top_k=5)

        # 4. 结构过滤：避免 downstream 解包失败
        valid_triples = [t for t in retrieved_triples if isinstance(t, (tuple, list)) and len(t) == 3]

        return valid_triples


    
    def memory_stats(self) -> Dict[str, float]:
        """获取记忆库统计信息"""
        if not self.memory_bank.memory_errors:
            return {
                'size': 0,
                'avg_error': 0,
                'min_error': 0,
                'max_error': 0,
            }
        
        return {
            'size': len(self.memory_bank.memory_vectors),
            'avg_error': sum(self.memory_bank.memory_errors) / len(self.memory_bank.memory_errors),
            'min_error': min(self.memory_bank.memory_errors),
            'max_error': max(self.memory_bank.memory_errors),
        }
    def load_memory(self, path_prefix="graph_memory"):
        """从 GMB 加载，并还原至运行时记忆库"""
        self.graph_memory.load_all(path_prefix)

        for edge in self.graph_memory.graph_edges:
            triple = (edge["from"], edge["label"], edge["to"])
            s_id = edge["from"]
            o_id = edge["to"]

            s_vec = self.graph_memory.graph_nodes.get(s_id, {}).get("vector", None)
            o_vec = self.graph_memory.graph_nodes.get(o_id, {}).get("vector", None)



            if s_vec and o_vec:
                s_tensor = torch.tensor(s_vec)
                o_tensor = torch.tensor(o_vec)
            if s_vec and o_vec:
                s_tensor = torch.tensor(s_vec)
                o_tensor = torch.tensor(o_vec)
    
            # 动态对齐维度：取最小公共长度（优先方案）
                min_dim = min(len(s_tensor), len(o_tensor))
                s_aligned = s_tensor[:min_dim]
                o_aligned = o_tensor[:min_dim]
                vector = (s_aligned + o_aligned) / 2
    
               # 投影变换替代裁剪/填充（当语义空间不一致时）
               # projection = nn.Linear(64, 384)  # 需预训练权重
               # o_projected = projection(o_tensor.float())
               # vector = (s_tensor + o_projected) / 2
    
                # 添加记忆（统一移至分支外）
                self.memory_bank.add_memory(
                    key=triple,
                    vector=vector.unsqueeze(0),
                    error=edge["error"]
                )
            
                


# ======================== 测试函数 ========================
def test_memory_agent():
    print("\n=== 🧪 空间增强记忆智能体测试 ===")
    
    # 创建智能体
    agent = PredictiveCodingAgent(
        tn_embed_dim=32,
        mps_bond_dim=16,
        mps_output_dim=64,
        hidden_dim=128,
        memory_capacity=50,
        memory_threshold=0.05
    )
    
    # 测试数据
    test_triples = [
        ("爱因斯坦", "提出", "相对论"),
        ("牛顿", "发现", "万有引力"),
        ("图灵", "发明", "图灵机")
    ]
    
    # 初始预测
    prediction, err = agent.forward(test_triples)
    print(f"✅ 初始预测误差: {err:.4f}")
    
    # 添加更多记忆
    for i in range(5):
        new_triple = (f"科学家{i}", f"发现{i}", f"理论{i}")
        agent.forward([new_triple])
        print(f"已添加记忆 {i+1}/5")
    
    # 测试记忆回忆
    query = ("爱因斯坦", "提出", "相对论")
    recalled = agent.recall_memory(query)
    print(f"\n🔍 回忆与查询相关的记忆:")
    for i, mem in enumerate(recalled[:3]):
        print(f"  记忆{i+1}: {mem}")
    
    # 获取记忆统计
    stats = agent.memory_stats()
    print(f"\n📊 记忆库统计:")
    print(f"  大小: {stats['size']}")
    print(f"  平均误差: {stats['avg_error']:.4f}")
    print(f"  最小误差: {stats['min_error']:.4f}")
    print(f"  最大误差: {stats['max_error']:.4f}")
    
    # 梯度测试
    target = torch.randn_like(prediction)
    loss = F.mse_loss(prediction, target)
    loss.backward()
    print("\n✅ 梯度反向传播成功")
    
    print("\n🧪 测试通过！")

if __name__ == "__main__":
    test_memory_agent()