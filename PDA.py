"""
🎯🎯 PredictiveDialogAgent.py
核心：以预测编码为 Agent 主控框架，整合提示词与 deepseek 接口，实现类人对话与记忆控制。
扩展：支持多轮思维链（CoT）管理，通过链式记录与回溯提升逻辑连贯性。
"""
import os
import torch
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from collections import deque
from typing import List, Tuple, Dict, Union
import asyncio
from PredictiveCoding import PredictiveCodingAgent
from GMB import GraphMemoryBank
from DSAP import MemoryAnchorUpdater
from MPR import MemoryRetriever


class DialogHistoryBuffer:
    def __init__(self, max_len=10):
        self.history = deque(maxlen=max_len)

    def add(self, user_input: str, agent_response: str):
        self.history.append((user_input, agent_response))

    def context_text(self) -> str:
        return "\n".join([f"你：{u}\nAI：{a}" for u, a in self.history])

    def thought_chain(self) -> str:
        return "\n".join([f"步骤{i+1}：用户提问：{u}\n        AI 回答：{a}" for i, (u, a) in enumerate(self.history)])


class PredictiveDialogAgent:
    def __init__(self, tool_threshold=0.25, memory_threshold=0.10, deepseek_api_key=None):
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.DSAP = MemoryAnchorUpdater()
        self.tool_trigger_threshold = tool_threshold
        self.memory_trigger_threshold = memory_threshold
        self.deepseek_api_key = deepseek_api_key

        self.dialog_buffer = DialogHistoryBuffer(max_len=10)

        gmb = GraphMemoryBank()

        self.pc = PredictiveCodingAgent(
            tn_embed_dim=32,
            mps_bond_dim=16,
            mps_output_dim=64,
            hidden_dim=128,
            memory_capacity=200,
            memory_threshold=memory_threshold,
            graph_memory_bank=gmb
        )

        self.pc.MPR = MemoryRetriever(memory_embeddings=[], memory_triples=[], similarity_threshold=0.7)

        self.client = AsyncOpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=deepseek_api_key
        )

    def save_memory_to_disk(self, path_prefix="dialog_memory"):
        self.pc.graph_memory.save_all(path_prefix)

    def load_memory_from_disk(self, path_prefix="dialog_memory"):
        if not os.path.exists(f"{path_prefix}.graph.json") or not os.path.exists(f"{path_prefix}.pt"):
            print(f"[⚠️] 未发现持久化记忆文件 '{path_prefix}', 跳过加载。")
            return
        self.pc.load_memory(path_prefix)
        print(f"[✅] 成功加载记忆：{path_prefix}")

    def embed(self, text: str) -> torch.Tensor:
        vec = self.embedder.encode(text, normalize_embeddings=True)
        return torch.tensor(vec).unsqueeze(0)

    def enhanced_memory_retrieval(self, query_text: str) -> List[Tuple[str, str, str]]:
        return self.pc.recall_memory(("查询", "内容", query_text))

    def get_memory_stats(self) -> Dict[str, float]:
        return self.pc.memory_stats()

    async def extract_triplet(self, text: str) -> Union[Tuple[str, str, str], None]:
        prompt = (
            f"请从下句中抽取出最重要的一组三元组（主体, 关系, 客体），"
            f"并以 [主体], [关系], [客体] 的格式返回，不要解释：\n{text}"
        )
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个信息抽取助手。"},
                    {"role": "user", "content": prompt}
                ]
            )
            content = response.choices[0].message.content.strip()
            if ',' in content:
                parts = [x.strip().strip('[]') for x in content.split(',')]
                if len(parts) == 3:
                    return tuple(parts)
        except Exception:
            return None
        return None

    async def chat(self, prompt: str) -> str:
        anchors_block = ""
        anchor_triples = []
        for k in self.DSAP.get_shared_anchors().keys():
            parts = k.split("::")
            if len(parts) == 3:
                anchor_triples.append((parts[0], parts[1], parts[2]))
        if anchor_triples:
            anchors_block = "\n你设置的身份/偏好关键点如下：\n" + "\n".join(
                [f"{s} — {p} — {o}" for s, p, o in anchor_triples])
            memory_embedding = self.pc.encode_input(anchor_triples)
        else:
            memory_embedding = torch.zeros(1, 64)

        context_block = f"\n以下是你与用户的最近几轮对话：\n{self.dialog_buffer.context_text()}" if self.dialog_buffer.history else ""
        chain_block = f"\n以下是你们的思维推理过程：\n{self.dialog_buffer.thought_chain()}" if self.dialog_buffer.history else ""

        knowledge_triples = self.enhanced_memory_retrieval(prompt)
        knowledge = "\n".join([f"{s} — {p} — {o}" for s, p, o in knowledge_triples]) or "（无相关记忆）"
        vec_summary = ", ".join([f"{v:.2f}" for v in memory_embedding[0].detach().numpy()[:5]])

        if knowledge_triples:
            _, avg_error = self.pc.forward(knowledge_triples)
            prediction_summary = f"预测编码误差: {avg_error:.3f}"
        else:
            avg_error = 0.0
            prediction_summary = "无预测编码数据"

        system_prompt = (
            "你是歌蕾蒂娅实验室的科研助手。\n"
            f"{anchors_block}{context_block}{chain_block}\n"
            f"以下是与输入相关的记忆内容：\n{knowledge}\n\n"
            f"向量摘要：{vec_summary}\n\n"
            f"记忆系统状态: {prediction_summary}\n\n"
            "【任务要求】请用自然语言回答用户问题，勿重复输出向量信息。\n"
            "如未找到相关知识，可逻辑推理。"
        )

        full_output = ""
        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )
            async for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                full_output += content
        except Exception as e:
            print(f"\n[❌] API 调用失败: {str(e)}")
            return "抱歉，处理请求时出现问题"

        _, err = self.pc.forward([("用户", "询问", prompt)])
        if err > self.memory_trigger_threshold:
            triple = await self.extract_triplet(prompt)
            if triple and any(triple):
                vec = self.embed(prompt)
                self.pc.graph_memory.add_triplet(triple, vec, error=err)
                print(f"\n📥 记忆更新：{triple}")

        self.dialog_buffer.add(prompt, full_output)
        return full_output

    def chat_with_user(self, user_input: str) -> str:
        return asyncio.run(self.chat(user_input))

    def clear_memory_systems(self):
        self.pc.graph_memory.graph_nodes.clear()
        self.pc.graph_memory.graph_edges.clear()
        self.dialog_buffer.history.clear()
        print("🧹 所有图结构记忆和对话历史已清空")

    def get_comprehensive_memory_stats(self) -> Dict[str, any]:
        stats = self.get_memory_stats()
        stats.update({
            'dialog_history_length': len(self.dialog_buffer.history),
            'anchor_count': len(self.DSAP.get_shared_anchors()),
        })
        return stats


# ==================== 启动入口 ====================
if __name__ == '__main__':
    print("=== 🧠 PDA 预测对话系统启动 ===")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "你的API_KEY")

    agent = PredictiveDialogAgent(deepseek_api_key=DEEPSEEK_API_KEY)
    agent.load_memory_from_disk("dialog_memory")

    async def main_loop():
        while True:
            q = input("\n你：")
            if q.strip().lower() in ["exit", "退出", "quit"]:
                break
            print("AI：", end="", flush=True)
            await agent.chat(q)

        print("\n💾 正在保存记忆...")
        agent.save_memory_to_disk("dialog_memory")

    asyncio.run(main_loop())
