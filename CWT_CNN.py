"""
📡 CWT-CNN Module for Agent Signal Reasoning
动态信号理解模块
Author: DOCTOR + 歌蕾蒂娅 (2025)
Description: This module enables AI agents to process dynamic 1D signals (e.g., sensor or behavioral rhythms)
via Continuous Wavelet Transform (CWT) and Convolutional Neural Networks (CNN).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pywt
import numpy as np



class CWTCNN(nn.Module):
    def __init__(self, input_size=(32, 128), output_dim=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * (input_size[0] // 4) * (input_size[1] // 4), output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return F.log_softmax(self.fc(x), dim=-1)


def generate_cwt_image(data, wavelet='morl', scales=np.arange(1, 64)):
    """
    使用 PyWavelets 执行连续小波变换并生成二维系数图像。
    """
    coefficients, freqs = pywt.cwt(data, scales, wavelet)
    return np.abs(coefficients)  # 返回时频图数据（可作为模型输入）


if __name__ == "__main__":
    # 🌊 模拟输入信号：正弦波 + 噪声
    t = np.linspace(0, 8 * np.pi, 128)
    test_signal = np.sin(t) + 0.3 * np.random.randn(128)

    # 🧠 CWT 变换
    cwt_map = generate_cwt_image(test_signal)
    input_tensor = torch.tensor(cwt_map[np.newaxis, np.newaxis, :, :], dtype=torch.float32)

    # 🤖 推理模型
    model = CWTCNN(input_size=cwt_map.shape, output_dim=3)
    result = model(input_tensor)
    print("📈 推理输出:", result)
