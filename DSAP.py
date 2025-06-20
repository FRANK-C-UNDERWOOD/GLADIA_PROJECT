"""
🧠 MemoryAnchorUpdater：锚点保护型记忆更新器
Author: DOCTOR + 歌蕾蒂娅 (2025)
Dirichlet stable anchor point
狄利克雷边界锚点
模块目的：提供一个稳定、可共享、可对抗遗忘的“核心记忆锚点”机制。
适用于多 Agent 系统，确保对话稳定性、身份一致性与知识图谱结构根节点不漂移。

🔧 模块功能概览：
1. MemoryAnchor：表示一个固定不变的三元组（如身份/偏好）及其嵌入。
2. MemoryAnchorUpdater：
   - add_anchor()      : 添加用户指定的锚点（如“我 是 DOCTOR”）
   - update_memory()   : 添加新记忆时避免锚点冲突或误更新
   - get_shared_anchors(): 返回所有 agent 共享的锚点信息
   - get_root_graph()  : 构建一个图结构以锚点为稳定根节点
"""

import torch
import time
from typing import Tuple, List, Dict

class MemoryAnchor:
    def __init__(self, triple: Tuple[str, str, str], embedding: torch.Tensor):
        self.triple = triple  # (subject, predicate, object)
        self.embedding = embedding  # 通常来自 encoder 的张量
        self.frozen = True  # 锚点不可更新
        self.created_at = time.time()  # 创建时间戳
        self.last_accessed = self.created_at  # 最后访问时间

    def touch(self):
        """更新最后访问时间"""
        self.last_accessed = time.time()

class MemoryAnchorUpdater:
    def __init__(self):
        self.anchors: Dict[Tuple[str, str, str], MemoryAnchor] = {}
        self.memory: List[Tuple[str, str, str]] = []
        # 共享缓存结构: {"triple_key": {"embedding": tensor, "timestamp": float}}
        self.shared_cache: Dict[str, Dict[str, object]] = {}
        self.version = "1.1"  # 版本号用于缓存兼容性检查

    def add_anchor(self, triple: Tuple[str, str, str], embedding: torch.Tensor):
        """添加锚点，并缓存供所有 Agent 调用"""
        if triple not in self.anchors:
            timestamp = time.time()
            anchor = MemoryAnchor(triple, embedding)
            self.anchors[triple] = anchor
            
            # 更新共享缓存，包含时间戳信息
            self.shared_cache["::".join(triple)] = {
                "embedding": embedding,
                "created_at": timestamp,
                "last_accessed": timestamp,
                "version": self.version
            }
            print(f"✅ 添加锚点：{triple} (创建于: {self.format_time(timestamp)})")
        else:
            # 更新访问时间但内容不变
            self.anchors[triple].touch()
            self.shared_cache["::".join(triple)]["last_accessed"] = time.time()
            print(f"⚠️ 锚点已存在：{triple} (最后访问: {self.format_time(self.anchors[triple].last_accessed)})")

    def update_memory(self, new_triples: List[Tuple[str, str, str]]):
        """
        添加新记忆时：
        - 避免覆盖已有锚点
        - 若三元组与锚点冲突，提示并跳过
        """
        for triple in new_triples:
            if triple in self.anchors:
                # 更新访问时间
                self.anchors[triple].touch()
                self.shared_cache["::".join(triple)]["last_accessed"] = time.time()
                print(f"🔒🔒 忽略锚点更新：{triple} (最后访问: {self.format_time(self.anchors[triple].last_accessed)})")
                continue
            self.memory.append(triple)
            print(f"📥📥 记忆新增：{triple} (时间: {self.format_time(time.time())})")

    def get_shared_anchors(self) -> Dict[str, torch.Tensor]:
        """用于其他 Agent 获取共享锚点嵌入（保持向后兼容）"""
        return {k: v["embedding"] for k, v in self.shared_cache.items()}
    
    def get_shared_anchors_with_metadata(self) -> Dict[str, Dict[str, object]]:
        """获取带元数据的共享锚点（包含时间戳）"""
        return self.shared_cache

    def get_root_graph(self) -> Dict[str, List[str]]:
        """
        构建以锚点为根节点的图结构（邻接表）
        所有 memory 三元组基于 subject -> object 组织
        添加时间戳信息到节点
        """
        graph = {}
        for triple, anchor in self.anchors.items():
            subj, _, obj = triple
            node_info = f"{obj} [锚点创建: {self.format_time(anchor.created_at)}]"
            graph.setdefault(subj, []).append(node_info)

        for triple in self.memory:
            subj, _, obj = triple
            node_info = f"{obj} [记忆添加: {self.format_time(time.time())}]"
            graph.setdefault(subj, []).append(node_info)

        return graph

    def save_shared_cache(self, path="anchor_cache.pt"):
        """保存带时间戳的缓存"""
        cache_data = {
            "metadata": {
                "version": self.version,
                "saved_at": time.time()
            },
            "anchors": self.shared_cache
        }
        torch.save(cache_data, path)
        print(f"💾 已保存共享缓存 (版本: {self.version})")

    def load_shared_cache(self, path="anchor_cache.pt"):
        """加载带时间戳的缓存"""
        cache_data = torch.load(path)
        
        # 版本兼容性检查
        if "metadata" in cache_data and cache_data["metadata"]["version"] != self.version:
            print(f"⚠️ 缓存版本不匹配: 当前版本 {self.version}, 缓存版本 {cache_data['metadata']['version']}")
        
        self.shared_cache = cache_data.get("anchors", {})
        
        for key, data in self.shared_cache.items():
            parts = key.split("::")
            if len(parts) == 3:
                triple = tuple(parts)
                anchor = MemoryAnchor(triple, data["embedding"])
                anchor.created_at = data.get("created_at", time.time())
                anchor.last_accessed = data.get("last_accessed", time.time())
                self.anchors[triple] = anchor
                print(f"🔍 加载锚点: {triple} (创建于: {self.format_time(anchor.created_at)})")
    
    def format_time(self, timestamp: float) -> str:
        """格式化时间戳为可读字符串"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

if __name__ == "__main__":
    updater = MemoryAnchorUpdater()
    emb = torch.randn(64)

    # 添加锚点：我 是 DOCTOR
    updater.add_anchor(("我", "是", "DOCTOR"), emb)
    time.sleep(1)  # 模拟时间流逝
    
    # 尝试添加记忆（包含锚点 + 新知识）
    updater.update_memory([
        ("我", "是", "DOCTOR"),  # 应跳过
        ("我", "喜欢", "Gladia"),
        ("DOCTOR", "身份", "博士")
    ])
    
    # 模拟再次访问锚点
    time.sleep(1)
    updater.add_anchor(("我", "是", "DOCTOR"), emb)  # 已存在锚点
    
    # 输出根节点图结构
    print("\n📊📊 稳定记忆图：")
    graph = updater.get_root_graph()
    for k, v in graph.items():
        print(f"{k}: {v}")
    
    # 测试保存和加载缓存
    updater.save_shared_cache("test_cache.pt")
    
    new_updater = MemoryAnchorUpdater()
    new_updater.load_shared_cache("test_cache.pt")
    
    print("\n🤝🤝 共享锚点元数据：")
    for k, v in new_updater.get_shared_anchors_with_metadata().items():
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v["created_at"]))
        accessed = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v["last_accessed"]))
        print(f"{k}: 创建于 {created}, 最后访问于 {accessed}")