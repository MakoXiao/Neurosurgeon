# 论文图表说明

本目录包含14张论文实验图表（PNG + PDF格式），以下为每张图的详细说明。

---

## Figure 1: 端到端延迟柱状图

**文件:** `figure1_latency_bar.{png,pdf}`

**内容:** 在LTE（10 MB/s）网络条件下，5个DNN模型（AlexNet、VGG-16、ResNet-18、MobileNet-V2、ResNet-50）分别使用7种推理策略的端到端延迟对比。每个子图对应一个模型，柱状图高度表示平均延迟（ms），误差棒表示标准差。ARL-Comp（紫色）柱状图使用加粗边框突出显示。

**论文用途:** 4.2.1节 延迟对比分析。直观展示ARL-Comp在所有模型上均优于Neurosurgeon，以及固定压缩基线（BL-0.3/0.5/0.7）的延迟-准确率权衡。

**关键数据:** ARL-Comp @10MB/s 延迟：AlexNet 15.1ms, VGG-16 56.9ms, ResNet-18 25.3ms, MobileNet-V2 37.4ms, ResNet-50 64.7ms。

---

## Figure 2: 分类准确率柱状图

**文件:** `figure2_accuracy_bar.{png,pdf}`

**内容:** 同样在LTE 10 MB/s条件下，7种方法的分类准确率（%）对比。All-Edge、All-Cloud、Neurosurgeon三者准确率相同（无压缩），ARL-Comp准确率接近基线，Baseline-0.3准确率下降最为明显。

**论文用途:** 4.2.2节 准确率分析。说明ARL-Comp将准确率损失控制在1.8%以内，而Baseline-0.3在MobileNet-V2上导致14.2%的准确率崩溃。

**关键数据:** MobileNet-V2准确率：基线71.0%, ARL-Comp 69.2%(-1.8%), BL-0.3 56.8%(-14.2%)。

---

## Figure 3: 延迟-带宽折线图（关键方法）

**文件:** `figure3_latency_vs_bandwidth.{png,pdf}`

**内容:** 5个关键方法（All-Edge、All-Cloud、Neurosurgeon、Baseline-0.5、ARL-Comp）的延迟随网络带宽（3G 5MB/s → WiFi 50MB/s）的变化趋势。X轴为对数刻度，每个子图对应一个模型。ARL-Comp使用加粗线条突出显示。

**论文用途:** 4.3节 网络带宽影响分析。展示ARL-Comp的带宽自适应特性——在所有带宽条件下均保持较低延迟，且在高带宽时能自动切换为All-Cloud策略。

**关键观察:** All-Edge延迟与带宽无关（水平线），All-Cloud延迟随带宽增大急剧下降，ARL-Comp在两者之间取得最优平衡。

---

## Figure 4: 准确率-带宽折线图（压缩方法）

**文件:** `figure4_accuracy_vs_bandwidth.{png,pdf}`

**内容:** 5种涉及分区/压缩的方法（Neurosurgeon、Baseline-0.3/0.5/0.7、ARL-Comp）的准确率随带宽变化的趋势。Neurosurgeon在所有带宽下准确率恒定（不压缩），ARL-Comp在高带宽下准确率回升至基线（因切换为All-Cloud，无需压缩）。

**论文用途:** 4.4节 压缩率影响分析。说明ARL-Comp在高带宽下自动选择不压缩策略，准确率完全恢复至基线水平。

**关键观察:** ARL-Comp准确率曲线在高带宽端向上弯曲，趋向Neurosurgeon水平，体现了自适应压缩的智能性。

---

## Figure 5: 准确率-延迟权衡散点图

**文件:** `figure5_accuracy_latency_tradeoff.{png,pdf}`

**内容:** 在LTE 10 MB/s下，每个模型中7种方法的（延迟, 准确率）二维分布。每个点代表一种方法，ARL-Comp（紫色星形）使用放大标记。理想位置为左上角（低延迟、高准确率）。

**论文用途:** 4.4.2节 帕累托最优性分析。直观展示ARL-Comp在准确率-延迟帕累托前沿上的位置——在维持高准确率的同时实现较低延迟。

**关键观察:** Baseline-0.3虽然延迟最低（最靠左），但准确率也最低（最靠下）；ARL-Comp位于右上区域，在两个维度上均优于大多数方法。

---

## Figure 6: ARL-Comp改善率柱状图

**文件:** `figure6_arl_improvement.{png,pdf}`

**内容:** ARL-Comp相对Neurosurgeon在4种带宽条件下的延迟降低比例（%）。每个子图对应一个模型，柱状图高度表示降低百分比，数值标注在柱顶。

**论文用途:** 4.3.3节 ARL-Comp相对Neurosurgeon的延迟降低。量化展示ARL-Comp在不同场景下的改善幅度。

**关键数据:** 最大改善：ResNet-50 @WiFi 77.3%；平均改善 @LTE约18%。

---

## Figure 7: 全方法延迟折线图

**文件:** `figure7_all_methods_latency_line.{png,pdf}`

**内容:** 全部7种方法的延迟随带宽变化的折线图。与Figure 3不同，此图包含所有方法（含Baseline-0.3/0.7），能更全面地观察各方法的带宽响应特性。

**论文用途:** 4.3节辅助图。供读者全面对比所有方法在不同带宽下的延迟表现，尤其是观察Baseline系列之间的梯度差异。

**关键观察:** 在低带宽端，各Baseline方法呈现出与压缩率成反比的延迟梯度（BL-0.3最低、BL-0.7最高）；在高带宽端，所有分区方法趋于收敛。

---

## Figure 8: 压缩率权衡分析图

**文件:** `figure8_compression_tradeoff.{png,pdf}`

**内容:** 双行图：上行为延迟 vs 压缩率（通道保留率1.0→0.3），下行为准确率 vs 压缩率。实线表示固定压缩基线（Neurosurgeon=1.0, BL-0.7, BL-0.5, BL-0.3），虚线表示ARL-Comp的对应值。每列对应一个模型。

**论文用途:** 4.4.1节 固定压缩率的延迟-准确率权衡。展示ARL-Comp如何在延迟和准确率两个维度上同时优于固定压缩方案——延迟接近BL-0.5水平，但准确率接近BL-0.7甚至Neurosurgeon水平。

**关键观察:** 上行图中ARL-Comp虚线低于固定压缩曲线的高压缩端（说明延迟更低），同时下行图中ARL-Comp虚线高于低压缩端（说明准确率更高）。

---

## Figure 9: 改善率热力图

**文件:** `figure9_heatmap_improvement.{png,pdf}`

**内容:** 5×4热力图，行为5个模型，列为4种带宽。单元格数值为ARL-Comp相对Neurosurgeon的延迟降低百分比（%）。颜色越深表示改善越大。

**论文用途:** 4.3.3节辅助图及4.6节综合讨论。一目了然地展示ARL-Comp在哪些模型-带宽组合下改善最为显著。

**关键观察:** 右下区域（大模型+高带宽）颜色最深，说明ARL-Comp对计算密集型模型在高带宽场景下优势最大（因自动切换为All-Cloud策略）。左上区域（小模型+低带宽）也有显著改善（因自适应选择小特征图分区点+适度压缩）。

---

## Figure 10: 各模型双轴深度分析图（×5）

**文件:**
- `figure10_AlexNet_dual_axis.{png,pdf}`
- `figure10_VGG_16_dual_axis.{png,pdf}`
- `figure10_ResNet_18_dual_axis.{png,pdf}`
- `figure10_MobileNet_V2_dual_axis.{png,pdf}`
- `figure10_ResNet_50_dual_axis.{png,pdf}`

**内容:** 为每个模型单独生成一张双Y轴折线图。左Y轴为延迟（ms，实线），右Y轴为准确率（%，虚线）。对比Neurosurgeon、Baseline-0.5和ARL-Comp三种方法随带宽的变化。

**论文用途:** 4.5节 逐模型深度分析。为每个模型提供独立的详细分析图，展示延迟和准确率随带宽的联合变化趋势。

**关键观察:** ARL-Comp的延迟曲线（实线）始终低于Neurosurgeon，而准确率曲线（虚线）始终高于Baseline-0.5，体现了"延迟更低、准确率更高"的双重优势。

---

## 图表编号与论文章节对应关系

| 图表 | 论文章节 | 主要说明内容 |
|------|---------|-------------|
| Figure 1 | 4.2.1 | 各方法延迟对比 |
| Figure 2 | 4.2.2 | 各方法准确率对比 |
| Figure 3 | 4.3 | 关键方法延迟-带宽趋势 |
| Figure 4 | 4.4 | 压缩方法准确率-带宽趋势 |
| Figure 5 | 4.4.2 | 准确率-延迟帕累托分析 |
| Figure 6 | 4.3.3 | ARL-Comp改善率量化 |
| Figure 7 | 4.3 | 全方法延迟-带宽趋势 |
| Figure 8 | 4.4.1 | 压缩率权衡分析 |
| Figure 9 | 4.3.3, 4.6 | 改善率全局热力图 |
| Figure 10 (×5) | 4.5 | 各模型双轴深度分析 |
