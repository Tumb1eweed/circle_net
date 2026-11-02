# 同一测试集上的加噪圆拟合对照

表格：[LaTeX 源码](figures/noise_fitting_table_with_module.tex) · [PDF](figures/noise_fitting_table_with_module.pdf) · [PNG](figures/noise_fitting_table_with_module.png)；[完整逐种子结果](results/all_fits_noise_test.json)。

这张表的所有**有数值**的行都来自本仓库同一个 HDF5 测试划分，而非论文表 3 的原数值。测试组为扫描组 10、11，共 162 个 `l0` patch，其中 160 个含圆。每档噪声采用 42–46 五个随机种子；每个 patch 的 XYZ 噪声标准差为该档百分比乘以原始 `l0` 包围盒对角线。`l0` 与 `l1` 加独立高斯噪声，真值标签保持不变。

| 方法 | 0.1% AD(c)代理 | 0.1% AD(r) | 0.1% MSE(r) | 0.5% AD(c)代理 | 0.5% AD(r) | 0.5% MSE(r) | 1% AD(c)代理 | 1% AD(r) | 1% MSE(r) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hyper | 0.001988 | 0.003817 | 0.000016 | 0.002197 | 0.003805 | 0.000016 | 0.007300 | 0.007052 | 0.000281 |
| LLS | **0.001957** | 0.004102 | 0.000019 | **0.002172** | 0.004149 | 0.000020 | 0.005414 | 0.006577 | 0.000070 |
| RANSAC | 0.003703 | 0.004174 | 0.000026 | 0.003016 | 0.004158 | 0.000025 | 0.005531 | 0.006083 | 0.000196 |
| RLTS（近似） | 0.002489 | 0.003967 | 0.000018 | 0.002875 | 0.004078 | 0.000021 | 0.005663 | 0.005792 | 0.000092 |
| PIE-NET | — | — | — | — | — | — | — | — | — |
| Ours（本地无 attention） | 0.004387 | 0.001636 | 0.000019 | 0.006880 | 0.003652 | 0.000094 | 0.014383 | 0.012111 | 0.000412 |
| Ours+Attn | 0.002747 | **0.001382** | **0.000003** | 0.002739 | **0.001284** | **0.000003** | **0.004540** | **0.002405** | **0.000017** |

三个指标都越小越好，粗体按**未四舍五入的五种子均值**逐列标记。`AD(c)` 是圆心误差代理：原始无额外噪声的 GT 边界点与 HDF5 法向拟合出的圆心作为参考，数据不含 CAD 圆心真值。距离单位是 **HDF5 原始坐标单位**，`MSE(r)` 是其平方；没有可靠的毫米换算。上表的 0.1%、0.5%、1% 噪声定义也不一定等于论文的噪声定义，所以这些值不能直接与论文表 3 的 mm/mm² 数值比较。

四种几何拟合都使用**同一个点间 attention 网络**预测的边界点（概率 ≥ 0.5，至少 10 个），并先以普通 PCA 投影到二维；神经网络行各用自己的软权重与预测法向拟合。Hyper 使用 [`circle-fit==0.2.1`](https://github.com/AlliedToasters/circle-fit) 的 `hyperSVD`；LLS 为线性代数最小二乘；RANSAC 为 256 次三点采样与内点 LLS 重拟合，阈值 `max(2σ, 10⁻⁶)`；RLTS 行是**自行实现的近似方案**，使用 70% 修剪、16 个随机三点初值、6 次 Hyper 迭代，并排除半径大于输入点二维包围盒对角线两倍的退化圆。它不是 RLTS 作者发布的精确实现。

每个随机种子都仅在上述六种已测方法的**共同有效正样本**上计算均值。三个噪声档的共同有效样本数分别为 **120.6/160、116.6/160、122.6/160**（五种子均值）；六种方法中，几何拟合的有效数均等于这个共同数，两个神经网络软拟合输出覆盖 160/160。该筛选会改变样本难度，所以这里的 Ours+Attn 数值也不同于先前对 160 个正样本计算的加噪结果。逐种子标准差与各方法覆盖数在 JSON 中。

PIE-NET 行留空：论文表 3 的 PIE-NET 是删去若干分支后在该论文的 patch-pair 数据上**重新训练**的模型，本仓库没有这一兼容检查点，[官方公开实现](https://github.com/wangxiaogang866/PIE-NET)也不是可直接读取本 HDF5 patch 的版本。将论文原数值或未适配的预训练结果填入本地对照表会混用实验条件。`Ours（本地无 attention）` 是本仓库的 `CircleNetV2` 60 轮检查点，不是论文作者的原始模型。

复现命令（`circlenet-repro` 环境）：

```bash
python eval_all_fits_noise.py --out results/all_fits_noise_test.json
python figures/render_all_fits_table.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=figures figures/noise_fitting_table_with_module.tex
gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=pngalpha -r200 -sOutputFile=figures/noise_fitting_table_with_module.png figures/noise_fitting_table_with_module.pdf
```
