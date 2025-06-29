项目简介
Predictive Dialog Agent（PDA） 是一个以预测编码为核心的对话智能体框架，旨在构建具备多轮记忆、类人推理能力、可控上下文管理和接口稳定性的智能对话系统。

本系统结合了结构化记忆（GMB）,记忆向量压缩(TN_MPS),语义检索（MPR）,动态状态控制（DSAP）等多个模块，支持并发调用,高效嵌入存储与回溯式对话追踪。

系统模块概览
1. GMB：Graph Memory Bank 图结构记忆库
支持实体–关系三元组的构建与检索。

实现符号型记忆存取与因果链条生成。

支持结构压缩与时间标注。

2.PC:Predictive Coding 预测编码系统
与反向传播与梯度下降等传统神经网络不同,PC模仿人脑的思维方式,模拟神经元的真实活动

3.TN_MPS 记忆向量压缩系统
提取三元组,经过TN_MPS的双层压缩后储存在图记忆库中.

4. DSAP：Dialog State & Action Planner 狄利特雷边界锚点
用于锚定对话内容不发生偏移和锚定核心记忆等功能。

支持思维重组、路径回溯、对话计划更新。

与 GMB 联动以生成上下文感知的响应策略。

5. MPR：Memory Projection Retriever 语义记忆召回器
基于向量检索（兼容 FAISS / 自定义存储结构）。

支持 embedding 维度动态配置（当前版本为 384d）。

与 MemoryRetriever 模块协作进行记忆匹配与去重。

6. MemoryRetriever 自定义向量召回模块
支持设定相似度阈值、动态权重调整。

可接入任意后端嵌入模型（如 SentenceTransformer、deepseek）。

检查维度一致性机制已接入，防止向量不匹配。

工程部署特性
支持并发：异步调用，兼容多会话请求。

接口稳定：模块化封装，提供统一 API 接口。

存储高效：向量与三元组分别存储，支持持久化与增量更新。

易调试：日志系统与错误捕获完善，便于追踪嵌入与召回异常。


启动方式
(没写main,直接run PDA文件就行,记得填上自己的deepseek api)
