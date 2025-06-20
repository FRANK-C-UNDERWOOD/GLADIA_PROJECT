"""
🧠 seRNN 模块：Spatially-Embedded Recurrent Neural Network
Author: DOCTOR + 歌蕾蒂娅 (2025)

模块用途：
- 在 RNN 中加入“神经元空间位置”作为连接结构限制
- 实现空间稀疏性约束，更符合生物神经网络的连接模式
- 可用于 Agent 空间导航记忆、脑连接模拟、图式记忆建构

主要组件：
1. seRNNCell       - 单个时间步的带空间惩罚的 RNN 单元
2. seRNN           - 多步序列建模的循环网络结构
3. spatial_regularizer - 连接距离正则项（用于加权 loss）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class seRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, neuron_coords: torch.Tensor):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.coords = neuron_coords  # shape: (hidden_size, 2)
        
        # 修复权重维度
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)  # 改为Linear层
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # 保存原始权重用于正则计算
        self.raw_W_hh = self.W_hh.weight  # (hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # 修复维度匹配问题
        h_linear = self.W_in(x) + self.W_hh(h_prev) + self.bias
        h_new = torch.tanh(h_linear)
        return h_new

    def spatial_regularizer(self):
        """修正正则项计算逻辑"""
        dist = torch.cdist(self.coords, self.coords, p=2)  # (N, N)
        # 点积计算：∑|W| * dist
        cost = torch.sum(torch.abs(self.raw_W_hh) * dist)
        return cost


class seRNN(nn.Module):
    def __init__(self, input_size, hidden_size, neuron_coords: torch.Tensor):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = seRNNCell(input_size, hidden_size, neuron_coords)

    def forward(self, x):
        B, T, D = x.shape
        # 添加设备信息
        device = x.device
        h = torch.zeros(B, self.hidden_size, device=device)
        h_seq = []
        
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            h_seq.append(h.unsqueeze(1))
            
        return torch.cat(h_seq, dim=1)

    def get_spatial_cost(self):
        return self.cell.spatial_regularizer()


if __name__ == "__main__":
    # 👾 示例：10维输入 → 16维空间神经元 → 序列长度 5
    coords = torch.rand(16, 2)  # 随机分布的神经元坐标
    model = seRNN(input_size=10, hidden_size=16, neuron_coords=coords)

    seq_input = torch.randn(4, 5, 10)  # (batch, time, input_dim)
    out = model(seq_input)
    print("🔮 输出 shape:", out.shape)
    print("📐 空间连接成本:", model.get_spatial_cost().item())
