"""
🧠 MPSMemoryEncoder: Agent Memory Representation with Matrix Product State (MPS)
Author: DOCTOR + 歌蕾蒂娅 (2025)
Description:
This module uses a linear tensor network structure (MPS) to encode sequences of semantic tokens
or user interaction features into a compressed and learnable memory embedding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class IntegratedTN_MPS(nn.Module):
    def __init__(self, 
                 tn_embed_dim=32,
                 mps_input_dim=3,
                 mps_bond_dim=16,
                 mps_output_dim=64):
        super().__init__()
        
        # 1. TN压缩模块
        self.tn_compressor = TripleCompressor(embed_dim=tn_embed_dim)
        
        # 2. 维度转换适配器 - 修改为更合适的结构
        self.adapter = nn.Sequential(
            nn.Linear(tn_embed_dim**3, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, tn_embed_dim * 3),  # 输出足够元素组成(s,p,o)序列
            nn.ReLU()
        )
        
        # 3. MPS记忆编码器
        self.mps_encoder = MPSMemoryEncoder(
            input_dim=mps_input_dim,
            feature_dim=tn_embed_dim,
            bond_dim=mps_bond_dim,
            output_dim=mps_output_dim
        )
        
        # 4. 配置参数
        self.tn_embed_dim = tn_embed_dim
        self.mps_input_dim = mps_input_dim
        
    def forward(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
        # 第一阶段：TN压缩
        batch_tensors = []
        for triple in triples:
            tn_tensor = self.tn_compressor.compress_triplet(triple)  # (D, D, D)
            flat_tensor = tn_tensor.flatten()  # (D³,)
            adapted = self.adapter(torch.tensor(flat_tensor).float())  # 输出为 (D*3)
            batch_tensors.append(adapted)
        
        # 构建MPS输入序列 (s,p,o)
        batch_size = len(triples)
        # 将列表堆叠并重塑为 (batch_size, seq_len=3, feature_dim)
        seq_input = torch.stack(batch_tensors).view(batch_size, self.mps_input_dim, self.tn_embed_dim)
        
        # 第二阶段：MPS记忆编码
        return self.mps_encoder(seq_input)

class TripleCompressor:
    """🔧 TN压缩器 - 保持不变"""
    def __init__(self, embed_dim=32):
        self.embed_dim = embed_dim
    
    def text_to_tensor(self, text: str) -> np.ndarray:
        byte_data = text.encode('utf-8', errors='replace')
        safe_length = min(len(byte_data), self.embed_dim)
        truncated = byte_data[:safe_length]
        vec = np.frombuffer(truncated, dtype=np.uint8)
        if len(vec) < self.embed_dim:
            vec = np.pad(vec, (0, self.embed_dim - len(vec)), constant_values=0)
        return vec.astype(np.float32)
    
    def compress_triplet(self, triple: Tuple[str, str, str]) -> np.ndarray:
        s_vec = self.text_to_tensor(triple[0])
        p_vec = self.text_to_tensor(triple[1])
        o_vec = self.text_to_tensor(triple[2])
        tensor = np.tensordot(s_vec, p_vec, axes=0)
        tensor = np.tensordot(tensor, o_vec, axes=0)
        return tensor

class MPSMemoryEncoder(nn.Module):
    """🧠 MPS编码器 - 保持不变"""
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        start = self.start.expand(B, -1).unsqueeze(1)
        result = torch.matmul(start, self._contract(x[:, 0, :], self.mps_cores[0]))
        for i in range(1, self.n_sites):
            result = torch.matmul(result, self._contract(x[:, i, :], self.mps_cores[i]))
        end = self.end.unsqueeze(0).expand(B, -1, -1)
        result = torch.matmul(result, end).view(B, 1)
        return self.fc(result)
    
    def _contract(self, feature_vec, mps_tensor):
        return torch.einsum('bf,dfr->bdr', feature_vec, mps_tensor)

# 修复后的测试函数
def test_integrated_system():
    print("\n=== 🧪 TN-MPS集成系统测试 (修复版) ===")
    
    # 1. 创建集成模型
    model = IntegratedTN_MPS(
        tn_embed_dim=32,
        mps_input_dim=3,
        mps_bond_dim=16,
        mps_output_dim=64
    )
    
    # 2. 测试数据
    test_triples = [
        ("爱因斯坦", "提出", "相对论"),
        ("牛顿", "发现", "万有引力"),
        ("图灵", "发明", "图灵机")
    ]
    
    # 3. 前向传播
    memory_vectors = model(test_triples)
    
    # 4. 验证输出
    print("✅ 输出形状:", memory_vectors.shape)  # 应为 (3, 64)
    print("✅ 数值范围: [{:.2f}, {:.2f}]".format(
        memory_vectors.min().item(),
        memory_vectors.max().item()
    ))
    
    # 5. 参数验证
    print("\n🔍 模块参数统计:")
    print(f"- TN压缩器: 嵌入维度={model.tn_embed_dim}")
    print(f"- MPS编码器: 输入维度={model.mps_input_dim}, 键维度={model.mps_encoder.bond_dim}")
    print(f"- 记忆维度: {memory_vectors.shape[-1]}")
    
    # 6. 梯度测试
    test_tensor = torch.randn(3, 64, requires_grad=True)
    loss = (memory_vectors - test_tensor).pow(2).mean()
    loss.backward()
    print("\n✅ 梯度反向传播成功")
    
    print("\n🧪 集成系统测试通过！")

if __name__ == "__main__":
    test_integrated_system()