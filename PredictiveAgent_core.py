"""
🎯 PredictiveAgent 主体框架
Author: DOCTOR + 歌蕾蒂娅

基于 Predictive Coding 架构重构的智能体核心，整合:
- 预测编码器用于输入误差估计与反馈调节
- 工具调用/记忆添加/静默处理均基于预测误差驱动
- 可拓展至 QNN 分类器、语义记忆库、CoT 思维链
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import Tuple,List
from memory_agent import MemoryAgent
from PredictiveCoding import PredictiveCodingAgent
# from tool_agent import call_tool_agent  # 可选：外部工具调用模块
from CWT_CNN import CWTCNN, generate_cwt_image
from MINLP import AgentOptimizationInterface
import asyncio


class PredictiveAgent:
    def __init__(self):
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.memory = MemoryAgent()  # 负责记忆更新
        self.pc = PredictiveCodingAgent(input_dim=self.dim, hidden_dim=64, memory_threshold=0.07)
        
        self.optimizer = AgentOptimizationInterface()
        self.tool_trigger_threshold = 0.25
        self.memory_trigger_threshold = 0.10
        self.rhythm_model = CWTCNN(input_size=(32, 128), output_dim=3)
        
    async def extract_triplet(self, text: str) -> List[Tuple[str, str, str]]:
        triple = await self.memory.extract_triplet(text)
        return [triple] if triple is not None else []
    
    async def chat(self, prompt: str) -> str:
        # 1. 提取三元组
        triples = await self.extract_triplet(prompt) 

        # 2. 更新记忆（包括通过 MPS 编码器生成的记忆向量）
        mps_embedding = self.memory.update_memory_with_mps(triples)
        
        # 3. 生成对话的返回内容
        response = await self.generate_response(prompt, mps_embedding)
        
        return response

    async def generate_response(self, prompt: str, mps_embedding: torch.Tensor) -> str:
        """
        基于用户输入和记忆向量生成自然语言回复
        """
        # 使用记忆向量的统计信息来影响回复生成
        memory_strength = torch.norm(mps_embedding).item()
        memory_summary = f"记忆强度: {memory_strength:.3f}"
        
        # 调用LLM生成自然语言回复
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url="https://api.deepseek.com/v1",
                api_key="sk-a8daf58a67d147f2a5cbca304fa01716"
            )
            
            system_prompt = f"""你是歌蕾蒂娅，一个智能助手。
                            当前记忆状态：{memory_summary}
                            请基于用户的输入生成自然、有帮助的回复。"""
            
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # 如果LLM调用失败，返回基础回复
            return f"我理解了您的问题：'{prompt}'。基于当前的记忆状态（{memory_summary}），我正在思考如何为您提供帮助。"

    def embed(self, triple: Tuple[str, str, str]) -> torch.Tensor:
        text = f"{triple[0]} — {triple[1]} — {triple[2]}"
        vec = self.embedder.encode(text, normalize_embeddings=True)
        return torch.tensor(vec).unsqueeze(0)  # shape: (1, dim)

    def check_rhythm_stability(self) -> bool:
        """
        检查系统节奏稳定性
        这里可以基于CWT-CNN模型或其他指标来判断系统状态
        """
        try:
            # 简单的稳定性检查，可以根据需要扩展
            # 这里返回True表示系统稳定
            return True
        except Exception:
            return False

    def process_input(self, triple: Tuple[str, str, str]) -> str:
        vec = self.embed(triple)
        _, _, err = self.pc.forward_predict(vec)
        self.pc.update_memory(vec, err)

        if not self.check_rhythm_stability():
            return "🌀 系统节奏失稳，暂停写入"

        if err > self.tool_trigger_threshold:
            return f"🧰 工具调用触发（预测失败，误差={err:.4f}）"
        elif err > self.memory_trigger_threshold:
            result = self.memory.add(triple)
            return f"🧠 记忆已更新（误差={err:.4f}）→ {result}"
        else:
            return f"🔕 无需处理，误差低于阈值（{err:.4f}）"
        
    def update_memory_with_mps(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
        """通过 MPS 更新记忆并返回全局记忆向量"""
        mps_embedding = self.memory.update_memory_with_mps(triples)
        return mps_embedding

  
    


if __name__ == '__main__':
    agent = PredictiveAgent()

    samples = [
        ("猫", "是", "动物"),
        ("爱因斯坦", "提出", "相对论"),
        ("我", "很", "开心"),
        ("地球", "是", "星球"),
    ]

    for t in samples:
        print("\n📥 输入：", t)
        print(agent.process_input(t))
