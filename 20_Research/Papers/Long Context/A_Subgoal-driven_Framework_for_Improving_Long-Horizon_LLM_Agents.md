---
title: "A Subgoal-driven Framework for Improving Long-Horizon LLM Agents"
authors: "Taiyi Wang, Sian Gooding, Florian Hartmann, Oriana Riva, Edward Grefenstette"
arxiv_id: "2603.19685"
pdf: "https://arxiv.org/pdf/2603.19685v1"
date_published: 2026-03
date_analyzed: 2026-03-23
categories: [cs.AI, cs.LG, cs.MA]
domain: "Long Context"
tags:
  - LLM-agents
  - web-navigation
  - reinforcement-learning
  - subgoal-decomposition
  - long-horizon-planning
  - milestone-rewards
---

# A Subgoal-driven Framework for Improving Long-Horizon LLM Agents

## 基本信息

| 属性 | 内容 |
|------|------|
| **作者** | Taiyi Wang, Sian Gooding, Florian Hartmann, Oriana Riva, Edward Grefenstette |
| **机构** | （基于作者背景推测）Google DeepMind / University College London |
| **arXiv** | [2603.19685](https://arxiv.org/abs/2603.19685) |
| **类别** | cs.AI, cs.LG, cs.MA |
| **发表日期** | 2026年3月 |

---

## 一句话总结

本文提出了一个基于子目标分解的智能体框架和一个基于里程碑奖励的强化学习训练框架 MiRA，显著提升了 LLM 智能体在长时间跨度网页导航任务中的表现。

---

## 研究动机与问题

### 核心问题

基于大语言模型（LLM）的智能体已成为数字环境（移动界面、操作系统、网页浏览器）中强大的自主控制器。然而，网页导航等任务需要处理动态内容和长序列操作，现有 LLM 智能体在长时间跨度规划方面面临两大核心挑战：

1. **在线执行中的迷失问题**：当新信息不断涌入时，智能体容易偏离目标，缺乏清晰且自适应的路径来达成最终目标。
2. **强化学习微调中的稀疏奖励问题**：延迟且稀疏的奖励信号使智能体难以识别哪些行动导致了成功，从而无法在扩展任务中保持连贯的推理。

### 研究背景

- 网页导航是一个特别具有挑战性的领域，需要处理动态页面内容和长序列决策
- 现有方法要么依赖端到端的 LLM 推理（容易在长任务中迷失），要么使用标准 RL 微调（受稀疏奖励限制）
- 长时间跨度任务中的信用分配问题（credit assignment）是 RL 训练的关键瓶颈

---

## 方法论

### 贡献一：基于子目标分解的在线规划框架

![子目标分解框架示意图](images/2603.19685_1.png)

该框架利用专有模型（如 Gemini）进行在线规划，核心思路是将复杂的长时间跨度任务分解为一系列可管理的子目标：

- **子目标生成**：利用强大的专有 LLM 将高层目标分解为有序的子目标序列
- **动态规划**：在执行过程中根据环境反馈实时调整子目标
- **自适应路径**：当新信息到达时，框架能够重新评估和更新当前的子目标计划
- **层次化执行**：智能体在每一步关注当前子目标，而非试图一次性解决整个问题

这种方法为智能体提供了清晰的中间目标，有效缓解了在长序列操作中"迷失方向"的问题。

### 贡献二：MiRA（Milestoning your Reinforcement Learning Enhanced Agent）

![MiRA 训练框架](images/2603.19685_2.png)

MiRA 是一个创新的 RL 训练框架，其核心设计理念是用密集的里程碑奖励信号替代传统的稀疏终端奖励：

- **里程碑定义**：将任务执行过程中的关键中间状态定义为里程碑（milestones）
- **密集奖励信号**：智能体在达到每个里程碑时获得即时奖励，而非仅在任务完成时获得反馈
- **改善信用分配**：密集奖励使智能体能够更准确地将成功归因于具体的行动序列
- **加速学习收敛**：通过提供更频繁的学习信号，加快 RL 微调过程

#### 与子目标框架的协同

MiRA 的里程碑机制与子目标分解框架形成互补：
- 推理时：子目标分解提供规划指导
- 训练时：里程碑奖励提供学习信号
- 两者共同解决长时间跨度任务中的规划和学习难题

---

## 实验结果

### 基准测试：WebArena-Lite

| 模型/方法 | 成功率 (SR) |
|-----------|------------|
| GPT-4o | 13.9% |
| GPT-4-Turbo | 17.6% |
| Gemini（基线） | ~X% |
| **Gemini + 子目标规划** | **~X+10%**（绝对提升约 10%） |
| WebRL（开源 SOTA） | 38.4% |
| Gemma3-12B（基线） | 6.4% |
| **Gemma3-12B + MiRA** | **43.0%** |

### 关键发现

![实验结果对比](images/2603.19685_3.png)

1. **子目标规划的效果**：在线子目标分解机制在 Gemini 等专有模型上带来约 **10% 的绝对成功率提升**
2. **MiRA 的显著提升**：将开源 Gemma3-12B 的成功率从 **6.4% 提升至 43.0%**，提升幅度高达 **36.6 个百分点**
3. **超越专有模型**：经 MiRA 训练的 Gemma3-12B（43.0%）显著超越了 GPT-4-Turbo（17.6%）和 GPT-4o（13.9%）
4. **超越开源 SOTA**：MiRA 的 43.0% 超过了此前开源模型最佳结果 WebRL 的 38.4%
5. **推理与训练的协同**：结合显式推理时规划与里程碑奖励可显著提升长时间跨度能力

---

## 技术细节分析

### 子目标分解的设计考量

- 子目标需要足够具体以指导行动，又足够抽象以允许灵活执行
- 动态重规划机制确保在环境状态变化时智能体能够适应
- 利用专有模型的强大推理能力进行高层规划，降低执行模型的负担

### MiRA 的奖励设计

- 里程碑的选择是关键——需要反映任务完成的有意义进展
- 密集奖励与稀疏终端奖励的平衡需要精心调节
- 里程碑奖励有效解决了长序列中的信用分配问题

### 模型规模与效率

- Gemma3-12B 作为一个相对较小的开源模型，经 MiRA 训练后即可超越大规模专有模型
- 这证明了训练方法比模型规模更重要的观点
- 为资源受限场景下部署高效网页导航智能体提供了可行方案

---

## 相关工作对比

| 方向 | 代表工作 | 本文优势 |
|------|---------|---------|
| LLM 网页智能体 | WebRL, Mind2Web | 结合子目标规划与密集奖励，性能更优 |
| 层次化规划 | SayCan, Inner Monologue | 专门针对网页导航的动态环境设计 |
| RL 微调 LLM | RLHF, DPO | 里程碑奖励替代稀疏奖励，适配长序列任务 |
| 子目标发现 | Option Framework, HRL | 与 LLM 推理能力结合，无需手动设计子目标 |

---

## 优势与局限

### 优势

- 框架设计清晰，子目标规划和 MiRA 可独立使用也可协同工作
- 在标准基准上取得了显著的性能提升
- 将开源小模型提升至超越大型专有模型的水平
- 方法具有通用性，可扩展至其他长时间跨度任务

### 潜在局限

- 子目标分解依赖强大的专有模型进行规划，增加了推理成本
- 里程碑的定义可能需要针对不同任务领域进行调整
- 评估主要集中在 WebArena-Lite 基准，在其他环境中的泛化能力有待验证
- 12B 参数模型虽然比专有模型小，但部署仍需一定计算资源

---

## 启发与思考

1. **子目标分解是解决长时间跨度问题的有效范式**：将复杂任务分解为可管理的子步骤，与人类解决问题的方式一致
2. **密集奖励信号对 RL 训练至关重要**：在长序列决策中，稀疏奖励是学习的主要瓶颈，里程碑奖励提供了一个优雅的解决方案
3. **推理时计算与训练时计算的互补**：本文展示了同时优化这两个维度的价值
4. **开源模型的潜力**：适当的训练方法可以使小型开源模型在特定任务上超越大型专有模型
5. **对未来自主智能体的启示**：更稳健和通用的自主系统需要同时具备良好的规划能力和高效的学习机制

---

## 关键术语

- **子目标分解 (Subgoal Decomposition)**：将长期目标分解为有序中间目标的策略
- **MiRA**：Milestoning your Reinforcement Learning Enhanced Agent，基于里程碑奖励的 RL 训练框架
- **里程碑奖励 (Milestone-based Rewards)**：在任务执行的关键中间节点提供的密集奖励信号
- **WebArena-Lite**：用于评估网页导航智能体的标准基准
- **长时间跨度规划 (Long-Horizon Planning)**：需要多步骤推理和执行的任务规划
- **信用分配 (Credit Assignment)**：在序列决策中确定哪些行动对最终结果有贡献的问题

---

## 引用

```bibtex
@article{wang2026subgoal,
  title={A Subgoal-driven Framework for Improving Long-Horizon LLM Agents},
  author={Wang, Taiyi and Gooding, Sian and Hartmann, Florian and Riva, Oriana and Grefenstette, Edward},
  journal={arXiv preprint arXiv:2603.19685},
  year={2026}
}
```
