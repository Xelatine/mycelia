# Mycelia（中文）

**一个会自发长连接的类脑系统，正在学习法语。**

Mycelia 不是层堆叠，而是一群小型自主节点组成的群落。它们通过消息交谈，自己
决定向谁发话；想联系谁就长出突触，长期无用的连接会被剪掉，再无信号到达的
节点会死亡，而当一条消息叫它时，它会以完全相同的权重复活。它与 Transformer
的赌注相反：智能也许来自**动态结构**（节点诞生、死亡、专门化），而不是单纯
的规模。生物原型是 FlyWire 绘制的果蝇大脑。

状态：六个阶段已全部实现、通过测试，并可在浏览器驾驶舱中实时观察。完整文档：
[README.md](../README.md)、[README.fr.md](README.fr.md)。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/train_tokenizer.py              # LeCarnet 语料 + 10 000 BPE
python mycelia/train.py --data text --scale 0  # 法语，10 000 步
python scripts/dashboard.py --train --data text --scale 0
# → http://127.0.0.1:8080
python -m pytest tests -q                      # 32 个测试
```

## 核心思想

* 宽度 32 的节点约 6 400 个参数；140 000 个节点是象征性上限，绝不使用稠密矩阵。
* 连接始终存为 `edge_index [2, 20 000]` 加计数器 `ne`，从不 N×N。
* 路由器使用查询/键与采样候选：每个地址 O(dim)，而不是 O(N)。
* 边在前向传播中诞生（每步最多 8 条）；死亡分两步，墓地保存精确权重。
* 驾驶舱实时显示不断接线的图、信号脉冲、七项生命体征，以及逐 token 的对话。
