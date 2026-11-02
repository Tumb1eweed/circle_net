# Circle-Net 后续实验记录（2026-11-02）

## 实际做的改动

- `train_v2.py` 按 81 块组成的扫描组划分数据；每组第 81 块作为全负的小块训练/评估样本。大块 `data_l1` 只作上下文，小块 `data_l0` 用于单圆参数预测。
- `model_v2.py` 用预测的边界概率和学习权重做可微圆拟合，并用网络预测平面法向。前 20 轮把真实边界掩码逐步退火到零；之后训练、验证和推理的拟合输入都不含真实边界标签。
- 分类损失从原始的正类频率反比加权交叉熵，改为平方根频率权重的交叉熵加 Dice；另外监督半径、法向和真实边界点到预测圆的残差。
- `model_direct.py` 是另一条分支：对预测边界加权池化，再由 MLP 直接输出圆心、半径和法向；推理时不运行圆拟合器。训练目标沿用上面的分类和几何约束。
- `model_point_attention.py` 在原骨干的逐点邻域特征后加入真正的点间注意力：500 个 `l0` 点按 Morton/Z-order 排序，两个 32 点局部窗口块交替错位，再用一个全局块让 500 点互相注意。之后仍使用可微圆拟合。它借鉴 PTv3 的序列化窗口思想，但不是官方 PTv3，也没有加载预训练权重。
- 仅保存固定第 60 轮的模型，没有按验证指标挑选检查点。

## 两种扫描组划分

| 协议 | 训练组 | 验证组 | 测试组 | 用途 |
|---|---|---|---|---|
| `stress` | 0,1,3,4,6,7,9,10 | 2,5 | 8,11 | 留出所有 `group % 3 == 2` 的模式，检验未见过的数据模式 |
| `balanced` | 0–7 | 8,9 | 10,11 | 训练组包含全部三种周期性模式，同时保持扫描组独立 |

数据中每三组呈现重复的点密度模式。例如背景点的第 16 近邻距离在组 0/1/2 的中位数分别约为 0.0190/0.0197/0.0206。不能仅根据 HDF5 断定这些模式对应什么物理扫描设置。

## 第 60 轮独立测试结果

| 模型和协议 | 逐点 F1 | 预测半径 MAE | 对真实边界拟合的参考半径 MAE | 预测圆心到参考拟合圆心的平均距离 |
|---|---:|---:|---:|---:|
| 可微拟合，`stress` | 0.295 | 0.04194 | 0.00419 | 0.01332 |
| 可微拟合，`balanced` | 0.799 | 0.00557 | 0.00407 | 0.00731 |
| 直接神经回归，`balanced` | 0.806 | 0.00177 | 0.00407 | 0.01476 |
| 点间 attention + 可微拟合，`balanced` | 0.842 | 0.00164 | 0.00407 | 0.00341 |

所有距离都是 HDF5 原始坐标单位，尚未确认毫米换算。直接回归的半径 MAE 很低，但圆心代理误差反而更大；它可能在学习八种标称半径的识别，而非更准确地恢复实测圆。圆心参考值由真实边界点拟合得到，不是 CAD 真值。测试集仅两个全负小块，负样本误报率不可据此作稳健结论。

`stress` 与 `balanced` 的模型结构和训练轮数相同，结果差异主要反映训练集是否覆盖测试中的周期性数据模式。其他损失和端到端拟合改动相对旧版可能有收益，但需要在同一划分下逐项消融，不能用这两个结果证明它们的单独效果。旧版按孔位置划分、同一扫描组进入训练和验证，也不能与这里的独立测试直接比较。

同一 `balanced` 划分、同一随机种子和 60 轮配置下，点间 attention 相比原两特征向量 attention 的测试 F1 从 0.799 提升到 0.842，软半径 MAE 从 0.00557 降到 0.00164。最小两类半径（编码 6/7）的 F1 从 0.510/0.469 提升到 0.710/0.667。硬阈值下至少 10 个预测边界点的覆盖率从 78.75% 降到 75.0%；参数误差均在全部正样本上统计。该结果只有一个随机种子、两个留出的扫描组；验证分类损失后期上升，说明概率校准和泛化仍需改进。

RTX 4090 上用 8 个样本、每个 500/2000 点做 20 次模型前向计时：原模型约 7.9 ms/批，点间 attention 约 8.8 ms/批；参数量分别为 209,094 和 615,366。计时不包含数据加载和圆拟合。

## 复现命令

在 `CircleNet/` 下运行，使用 `circlenet-repro` 环境：

```bash
python train_v2.py --epochs 60 --split-protocol balanced --model fit --out-dir log/v2_balanced_20261102
python train_v2.py --epochs 60 --split-protocol balanced --model direct --out-dir log/direct_balanced_20261102
python train_v2.py --epochs 60 --split-protocol balanced --model point_attention --out-dir log/point_attention_balanced_20261102
python eval_v2.py --checkpoint log/v2_balanced_20261102/last_model.pth --split test
python eval_v2.py --checkpoint log/direct_balanced_20261102/last_model.pth --split test
python eval_v2.py --checkpoint log/point_attention_balanced_20261102/last_model.pth --split test --by-radius
python plot_v2.py --metrics log/v2_balanced_20261102/metrics.jsonl --out log/v2_balanced_20261102/training_curves.png
```

官方 PTv3/Sonata 包和预训练权重未纳入上述结果；点间 attention 分支为针对当前 500 点任务实现的轻量对照。
