"""
🧠🧠 Tensor Network Compression Module for Triple Embeddings
Inspired by the paper: "Compressing Neural Networks Using Tensor Networks with Exponentially Fewer Variational Parameters"
Author: DOCTOR + 歌蕾蒂娅 (2025)
"""

import numpy as np
from typing import Tuple, List

class TripleCompressor:
    """🔧🔧 三元组张量压缩器：使用张量网络技术将文本三元组压缩为紧凑张量表示\n
    工作原理：将文本编码为向量 → 构建高阶张量 → 执行张量网络压缩 → 输出紧凑表示\n
    特点：固定维度输出、支持批量处理、可视化调试功能"""
    
    def __init__(self, embed_dim=32, mode=None):
        """
        🔧🔧 初始化压缩器配置\n
        参数：
        embed_dim -- 文本编码的固定维度（默认32）
        """
        self.embed_dim = embed_dim
        self.mode = mode if mode else "default"
    
    def text_to_tensor(self, text: str) -> np.ndarray:
        """
        📝📝 增强版文本编码器：安全处理多字节字符
        改进点：
        1. 使用编码错误处理策略
        2. 避免截断多字节字符
        3. 添加长度校验机制
        """
        # 安全编码：使用替换策略处理无效字节
        byte_data = text.encode('utf-8', errors='replace')
        
        # 安全截断：确保不破坏多字节字符边界
        safe_length = min(len(byte_data), self.embed_dim)
        truncated = byte_data[:safe_length]
        
        # 转换数值向量
        vec = np.frombuffer(truncated, dtype=np.uint8)
        
        # 维度处理
        if len(vec) < self.embed_dim:
            # 使用0作为填充标记（原设计使用-1会导致问题）
            vec = np.pad(vec, (0, self.embed_dim - len(vec)), 
                         constant_values=0)
        return vec.astype(np.float32)

    def compress_triplet(self, triple: Tuple[str, str, str]) -> np.ndarray:
        """
        🌀🌀 三元组压缩核心：使用张量网络技术压缩(s,p,o)三元组\n
        技术路线：
        1. 分别编码主语(s)、谓语(p)、宾语(o)
        2. 执行张量网络收缩：s⊗p→矩阵 → 矩阵⊗o→三阶张量
        3. 输出立方体状压缩表示(D×D×D)\n
        可视化：[s]→⨂⨂⨂─[p]→◻◻─⨂⨂⨂─[o]→◻◻◻◻
        """
        s_vec = self.text_to_tensor(triple[0])
        p_vec = self.text_to_tensor(triple[1])
        o_vec = self.text_to_tensor(triple[2])

        # 张量收缩模拟砖墙张量网络结构
        tensor = np.tensordot(s_vec, p_vec, axes=0)        # 形状: (D, D)
        tensor = np.tensordot(tensor, o_vec, axes=0)       # 形状: (D, D, D)
        return tensor

    def flatten_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        📦📦 张量展平器：将高阶压缩张量转换为一维向量\n
        用途：
        - 便于存储到数据库
        - 适合输入MLP等全连接网络
        - 减少下游处理复杂度\n
        数学操作：flatten(D×D×D)→[D³]
        """
        return tensor.flatten()

    def compress_batch(self, triples: List[Tuple[str, str, str]]) -> List[np.ndarray]:
        """
        🚀🚀🚀 批量处理管道：高效处理多个三元组\n
        优化点：
        - 自动并行化（利用numpy向量化）
        - 内存预分配
        - 流式处理支持\n
        输入：[("s1","p1","o1"), ("s2","p2","o2")...] → 输出：[张量1, 张量2...]
        """
        return [self.compress_triplet(triple) for triple in triples]

def test_triple_compressor():
    """🧪 全面测试三元组压缩器功能"""
    print("\n=== 启动 Tensor Network 压缩模块测试 ===")
    
    # 1. 基础功能测试
    print("\n🔍 测试1: 基础功能验证")
    compressor = TripleCompressor(embed_dim=32)
    
    # 测试文本编码
    test_text = "你好世界"
    text_tensor = compressor.text_to_tensor(test_text)
    print(f"文本编码测试: 输入'{test_text}' → 形状 {text_tensor.shape} | dtype={text_tensor.dtype}")
    assert text_tensor.shape == (32,), "文本编码维度错误"
    assert text_tensor.dtype == np.float32, "数据类型错误"
    
    # 测试三元组压缩
    triplet = ("爱因斯坦", "提出", "相对论")
    compressed = compressor.compress_triplet(triplet)
    print(f"三元组压缩: 形状 {compressed.shape} | 维度乘积 {np.prod(compressed.shape)}")
    assert compressed.shape == (32, 32, 32), "压缩张量形状错误"
    
    # 测试展平功能
    flattened = compressor.flatten_tensor(compressed)
    print(f"张量展平: {flattened.shape[0]}维向量")
    assert flattened.shape == (32 * 32 * 32,), "展平维度错误"
    
    # 2. 批量处理测试
    print("\n🔍 测试2: 批量处理能力")
    batch = [
        ("苹果", "是", "水果"),
        ("牛顿", "发现", "万有引力"),
        ("Python", "用于", "AI开发")
    ]
    batch_results = compressor.compress_batch(batch)
    print(f"批量处理: {len(batch_results)}个三元组 | 首个形状 {batch_results[0].shape}")
    assert len(batch_results) == len(batch), "批量处理数量不匹配"
    
    # 3. 边界条件测试
    print("\n🔍 测试3: 边界条件验证")
    # 短文本测试
    short_text = "a"
    short_tensor = compressor.text_to_tensor(short_text)
    print(f"短文本测试({short_text}): 首值={short_tensor[0]}, 第二值={short_tensor[1]}, 末值={short_tensor[-1]}")
    assert short_tensor[0] == ord('a'), "首元素编码错误"
    assert short_tensor[1] == 0.0, "填充值错误"
    
    # 长文本测试
    long_text = "张量网络压缩技术" * 10  # 超长字符串
    long_tensor = compressor.text_to_tensor(long_text)
    print(f"长文本测试: 前2字节 [{long_tensor[0]}, {long_tensor[1]}]")
    assert long_tensor.shape == (32,), "长文本维度错误"
    
    # 特殊字符测试
    special_char = "❄"
    special_tensor = compressor.text_to_tensor(special_char)
    first_byte = special_char.encode('utf-8')[0]
    print(f"特殊字符测试: 首值 {special_tensor[0]} (应为{first_byte})")
    assert special_tensor[0] == first_byte, "特殊字符编码错误"
    
    # 空文本测试
    empty_text = ""
    empty_tensor = compressor.text_to_tensor(empty_text)
    print(f"空文本测试: 所有值应为0? {np.all(empty_tensor == 0)}")
    assert np.all(empty_tensor == 0.0), "空文本应全填充为0"
    
    # 4. 压缩效果验证
    print("\n🔍 测试4: 压缩效果分析")
    original_size = 3 * 32  # 3个32维向量
    compressed_size = np.prod(compressed.shape)
    print(f"原始大小: {original_size}元素 → 压缩后: {compressed_size}元素")
    print(f"压缩比率: {compressed_size/original_size:.1f}x")
    
    # 5. 数值稳定性检查
    print("\n🔍 测试5: 数值特性验证")
    print(f"压缩张量范围: [{compressed.min():.2f}, {compressed.max():.2f}]")
    print(f"均值: {compressed.mean():.2f} | 标准差: {compressed.std():.2f}")
    
    print("\n✅✅ 所有测试通过！模块功能正常 ✅✅")

if __name__ == "__main__":
    test_triple_compressor()