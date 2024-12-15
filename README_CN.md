# 增强型方向连接医学图像分割模型

[English](README.md) | [中文](README_CN.md)

本仓库包含了我们对DconnNet的增强版本，用于医学图像分割任务。该工作基于CVPR2023论文"基于方向连接的医学图像分割" [[论文](https://arxiv.org/pdf/2304.00145.pdf)]。

## 主要创新与改进分析

我们的增强版本在原始DconnNet的基础上进行了全面的改进，特别适合处理具有复杂解剖结构的医学图像，我们的改进主要包括：

### 改进的 Decoder Attention 机制

---

在解码阶段，我们动态加权融合不同层级的特征图，提升了特征交互和分割结果的细节表现。

---

#### 改进方法

1. **动态特征融合：**  
  将解码路径中不同分辨率的特征图（$c_1, c_2, c_3, c_4$）通过拼接形成联合特征：
  $$F_{\text{concat}} = \text{Concat}(c_1, c_2, c_3, c_4).$$

  使用 $1 \times 1$ 卷积和 $3 \times 3$ 卷积提取全局和局部上下文信息，生成动态权重特征图：
  
  $$F_{\text{attention}} = \text{Conv}_{1 \times 1}(\text{ReLU}(\text{Conv}_{3 \times 3}(F_{\text{concat}}))).$$

2. **权重加权融合：**  
  动态加权机制将各层特征进行加权融合，生成最终解码输入：
  $$F_{\text{fused}} = \sum_{i=1}^4 \alpha_i \cdot c_i, \quad \alpha_i \text{由} F_{\text{attention}} \text{确定}.$$

---

#### 改进点分析

1. **增强跨层级特征交互：**  
   动态加权机制使特征融合更具选择性，有助于捕获多分辨率上下文信息。
   
2. **轻量化：**  
   模块仅使用简单卷积，计算复杂度低，易于集成，并不会增加模型复杂度。

---

通过改进，模型在保持计算效率的同时，实现了对不同分辨率特征的动态加权融合，有效提升了分割结果的一致性和精确性。

### 使用 MRDE 模块替代原有的 SDE 模块

---

MRDE 模块通过多分辨率架构、轻量化卷积和更稳定的方向增强策略，不仅在计算效率上显著优于 SDE，还能更好地保持梯度稳定性和方向信息的提取能力。

---

#### 1. 原 SDE 模块

##### 1.1 模块结构
SDE 模块的设计包括三个关键部分：**方向先验提取**、**通道分割**和**子路径激励**。其流程如下：

1. **方向先验提取 (Directional Prior Extraction)**  
   SDE 利用方向信息嵌入的全局特征 $e_5$ 生成方向先验 $\alpha_{\text{prior}}$。具体过程为：
   - $e_5$ 被上采样到输入尺寸，产生方向性输出 $X_{\text{prior}}$：
     $$X_{\text{prior}} = \text{Conv}(\text{Upsample}(e_5)).$$
   - 使用全局平均池化 (GAP) 和 $1 \times 1$ 卷积层提取方向嵌入 $v_{\text{prior}}$：
     $$v_{\text{prior}} = \delta(W_1 \cdot \text{GAP}(X_{\text{prior}})),$$
     其中 $\delta$ 是 ReLU 激活函数。
   - 方向先验 $\alpha_{\text{prior}}$ 通过一个 Sigmoid 函数归一化：
     $$\alpha_{\text{prior}} = \sigma(W_2 \cdot v_{\text{prior}}).$$

2. **通道分割 (Channel-wise Slicing)**  
   $e_5$ 和 $\alpha_{\text{prior}}$ 被沿通道维度切分为 $i$ 组子通道 $e_5^i$ 和 $\alpha_{\text{prior}}^i$，每组负责建模不同方向的特征。

3. **子路径激励 (Sub-path Excitation, SPE)**  
   每组子通道 $e_5^i$ 通过双重注意力模块 (PAM 和 CAM) 提取上下文相关性和通道间依赖性：
   $$e_5^{i'} = \text{PAM}(\text{CAM}(e_5^i)).$$
   使用方向先验 $\alpha_{\text{prior}}^i$ 对特征进行选择性激活：
   $$e_{\text{SDE}}^i = W_3^i (\alpha_{\text{prior}}^i \cdot e_5^{i'}) + e_5^i.$$
   最后，各子路径的输出堆叠并重编码为新特征 $e_{\text{SDE}}$。

---

##### 1.2 存在的不足
1. **单分辨率特征提取：**  
   SDE 模块仅在单一分辨率 $e_5$ 上提取方向特征，难以同时捕获全局拓扑信息和局部细节。

2. **计算复杂度高：**  
   每个子路径中使用了独立的 PAM 和 CAM，导致计算和内存开销较高。

3. **梯度不稳定：**  
   SDE 模块在多路径并行计算中，梯度传递路径过长，容易引发梯度爆炸或消失问题。

---

#### 2. 改进后的 MRDE 模块

##### 2.1 模块结构
MRDE 模块的设计以多分辨率增强为核心，分为以下关键部分：

1. **多分辨率金字塔特征提取：**  
   输入特征 $X$ 被下采样至多种分辨率 $\{1, 2, 4\}$，生成不同尺度的特征 $X_s$：
   $$X_s = \text{AdaptiveAvgPool}(X, (H/s, W/s)), \quad s \in \{1, 2, 4\}.$$

2. **方向卷积增强 (Directional ConvBlock)：**  
   每个尺度特征 $X_s$ 通过方向卷积模块处理，提取方向信息：
   $$X_s' = \text{DepthwiseConv}(X_s) + \text{PointwiseConv}(X_s).$$

3. **多分辨率特征融合：**  
   所有尺度的方向增强特征 $\{X_1', X_2', X_4'\}$ 被上采样至统一尺寸并拼接：
   $$X' = \text{Concat}([X_1', X_2', X_4'], \text{dim}=C).$$

4. **注意力机制优化：**  
   - **通道注意力：** 通过全局平均池化和 MLP 生成通道权重 $\alpha_{\text{channel}}$：
     $$\alpha_{\text{channel}} = \sigma(\text{MLP}(\text{GAP}(X'))).$$
   - **空间注意力：** 通过 $7 \times 7$ 卷积生成空间权重 $\alpha_{\text{spatial}}$：

     $$\alpha_{\text{spatial}} = \sigma(\text{Conv}_{7 \times 7}(X')).$$
  
   最终增强特征：

   $$X_{\text{MRDE}} = X' \cdot \alpha_{\text{channel}} \cdot \alpha_{\text{spatial}}.$$

5. **归一化与残差连接：**  
   输出特征通过 L2 归一化和残差路径保持稳定：
   $$X_{\text{output}} = \frac{X_{\text{MRDE}}}{\|X_{\text{MRDE}}\|_2 + \epsilon}.$$

---

##### 2.2 改进点分析
1. **多分辨率特征捕获：**  
   MRDE 模块能在多尺度下同时提取全局和局部特征，相较于 SDE 的单尺度处理，更适用于捕获复杂结构。

2. **轻量化设计：**  
   MRDE 使用深度可分离卷积代替 SDE 中的 PAM 和 CAM，大幅降低了计算复杂度。

  - **SDE 模块：**  
    输入大小为 $C \times H \times W$，SDE 的复杂度为：
    $$\mathcal{O}_{\text{SDE}} = 8 \cdot (C^2HW + CHW) \approx 8C^2HW.$$
  - **MRDE 模块：**  
    MRDE 的多分辨率处理复杂度为：
    
    $$\mathcal{O}_{\text{MRDE}} = \sum_{s \in \{1, 2, 4\}} \left( C^2H^2/s^2 \right) \approx 1.75C^2HW.$$

3. **梯度稳定性：**  
   MRDE 在每个尺度中独立计算方向特征，减少了梯度传递路径，结合输入输出归一化避免梯度爆炸。
  - **SDE 模块：**  
    子路径间梯度独立计算导致较高的不稳定性：
    $$\nabla L_{e_5^i} = \nabla L_{\text{concat}} \cdot \frac{\partial \text{PAM}(\text{CAM}(e_5^i))}{\partial e_5^i}.$$
  - **MRDE 模块：**  
    MRDE 的归一化和方向卷积减少了梯度波动。

---

MRDE 模块在 SDE 的基础上，通过多分辨率方向增强设计，显著提升了方向特征提取的精确性和效率，具备更好的梯度稳定性和鲁棒性。这使得 MRDE 模块在医学图像分割任务中表现出更优的拓扑一致性和计算效率，特别适用于复杂结构和高分辨率输入。

### 使用 GLFI 模块增强全局与局部特征交互

---

GLFI 模块通过结合多尺度全局特征提取、边缘增强和图注意力机制，在捕获复杂结构和优化边界细节方面实现了显著提升，并在全局-局部特征交互中提供了更加鲁棒和高效的解决方案。此模块被集成到 DconnNet 的解码路径中，作为特征细化与融合的关键步骤。

---

#### 1. 模块结构设计

GLFI（Global-Local Feature Interaction）模块旨在通过多视角特征建模来增强全局和局部特征的交互。其设计包括以下核心部分：

1. **多尺度全局特征提取**  
   为了捕获不同感受野下的全局上下文信息，GLFI 模块通过四个空洞卷积分支提取全局特征：

   $$F_{\text{global}} = \text{Concat}\left[\text{DilatedConv}_{r=1}, \text{DilatedConv}_{r=2}, \text{DilatedConv}_{r=4}, \text{DilatedConv}_{r=8}\right],$$
   
   其中 $r$ 为空洞卷积的扩张率。每个分支的输出特征代表特定尺度的全局信息，并在通道维度上拼接以形成多尺度全局特征表示。

2. **边缘检测增强**  
   使用增强的 Sobel 算子提取边缘特征，并通过可学习卷积模块细化这些边缘信息：
   $$F_{\text{edge}} = \text{Refine}(\text{Sobel}_x(F) \oplus \text{Sobel}_y(F)),$$
   其中 $\text{Sobel}_x$ 和 $\text{Sobel}_y$ 分别为 X 和 Y 方向的 Sobel 滤波，$\oplus$ 表示特征拼接，$\text{Refine}$ 为用于细化边缘特征的卷积网络。通过这种方式，GLFI 能够增强模型对边界和局部细节的感知能力。

3. **图注意力特征细化**  
   在全局与局部特征的交互过程中，使用多头图注意力机制建模特征间的全局关系。具体步骤包括：
   - 线性变换生成查询 ($Q$)、键 ($K$)、值 ($V$)：
     $$Q = W_q F, \quad K = W_k F, \quad V = W_v F,$$
     其中 $W_q, W_k, W_v$ 分别为线性变换矩阵。
   - 计算注意力权重：
     $$A = \text{Softmax}\left(\frac{Q \cdot K^\top}{\sqrt{d_k}}\right),$$
     其中 $d_k$ 是缩放因子，用于稳定梯度。
   - 基于注意力权重和特征值 $V$ 生成注意力特征：
     $$F_{\text{attn}} = A \cdot V.$$
   - 投影输出：
     $$F_{\text{out}} = W_o F_{\text{attn}},$$
     其中 $W_o$ 为输出投影矩阵。

4. **特征融合与优化**  
   将全局特征 $F_{\text{global}}$、边缘特征 $F_{\text{edge}}$ 和图注意力特征 $F_{\text{attn}}$ 进行融合，通过通道和空间注意力优化特征交互：
   $$F_{\text{fusion}} = \alpha_{\text{channel}} \cdot F_{\text{global}} + \alpha_{\text{spatial}} \cdot F_{\text{attn}},$$
   其中 $\alpha_{\text{channel}}, \alpha_{\text{spatial}}$ 分别通过全局平均池化和卷积生成的注意力权重，确保特征融合的有效性。

---

#### 2. 在模型中的应用

GLFI 模块被集成到 DconnNet 的解码路径中，主要作用于中间层的特征细化和融合。具体应用流程如下：
1. **多尺度特征提取后处理：**  
  在 DconnNet 解码路径中的每一层 ($d_4, d_3, d_2, d_1$)，通过 GLFI 模块细化上下文特征，第 $k$ 层解码特征 $d_k$ 会被输入到 GLFI 模块，用于增强局部边界和全局连贯性。

2. **多分辨率特征融合：**  
  编码路径的对应跳跃连接特征（通过 ResNet 编码器生成的多分辨率特征，也就是 $c_4, c_3, c_2, c_1$）将与 GLFI 处理后的解码特征进行融合，将编码路径的全局语义信息与解码路径的局部重建信息结合，生成更精细化的分割特征表示。多分辨率特征在每一个解码层级中得到进一步优化，确保模型能够同时捕获细节和全局拓扑结构。

3. **最终分割优化：**  
  在完成每一层解码器特征的融合与细化后，这些经过 GLFI 模块处理的层级特征会被统一输入到最终解码器模块中。最终解码器模块采用一个轻量化设计，接收各层级的融合特征（如 $d_4^{\text{GLFI}}, d_3^{\text{GLFI}}, d_2^{\text{GLFI}}, d_1^{\text{GLFI}}$），通过逐级上采样和特征融合操作生成全分辨率的分割结果。最终分割结果综合了所有层级的特征优势，能够在细小结构（如血管）和全局语义一致性（如器官轮廓）上表现更优。通过这种层级化的特征融合与优化，模型有效地提升了分割的精确性和鲁棒性，同时保证了对复杂边界和多尺度结构的处理能力。

---

#### 3. 改进点分析

1. **多尺度建模的全局特征**  
   通过不同扩张率的空洞卷积，GLFI 模块能够捕获丰富的多尺度全局上下文信息，更适合处理复杂结构和长距离依赖关系的分割任务。

2. **增强的边缘检测能力**  
   使用 Sobel 算子和可学习卷积细化边缘信息，能够显著提高模型对复杂边界和细小目标的敏感性，尤其在医学图像中表现出色。

3. **图注意力机制的全局依赖建模**  
   相较于传统的通道或空间注意力机制，图注意力机制能够更有效地在全局范围内建模特征之间的复杂依赖关系，从而提升语义一致性和拓扑结构的保留能力。

4. **融合优化的轻量化设计**  
   在保证特征表达能力的同时，通过注意力机制减少冗余计算，使模块在高分辨率输入下依然保持良好的效率。

---

GLFI 模块通过整合全局与局部特征的交互设计，为医学图像分割任务提供了更强的语义理解和边界细化能力。通过多尺度全局特征提取、边缘增强和图注意力机制的协同优化，GLFI 显著提升了分割结果的拓扑一致性和全局连贯性，为高分辨率和复杂结构场景的分割任务提供了有效的解决方案。

### 在原有的损失函数的基础上扩展改进损失函数

改进的损失函数通过引入多尺度频域损失、拓扑保留损失和分布式权重设计，不仅改善了模型对不平衡数据的鲁棒性，还显著增强了解剖结构的连贯性和方向一致性。

#### 1. 原有损失函数

##### 1.1 原有损失函数结构分析

原总损失函数的定义如下：
$$L_{\text{total}} = L_{\text{main}} + 0.3 \cdot L_{\text{prior}},$$
其中 $L_{\text{main}}$ 和 $L_{\text{prior}}$ 分别对应模型的主输出和来自 SDE 模块的辅助输出。这两个部分分别由大小密度损失（Size Density Loss, SDL）和双向连通性损失（Bicon Loss）组成：
$$L = L_{\text{SD}} + L_{\text{Bicon}}.$$

- 大小密度损失 $L_{\text{SDL}}$：
  SDL 针对医学数据的不平衡问题，引入基于标签大小分布的加权机制。首先，计算所有训练数据中每个类别的标签大小分布概率密度函数 $PDF_j(k)$，然后对每个样本的标签大小 $k$ 计算对应的加权系数 $P_j(k)$：
  
  $$P_j(k) = \begin{cases} 1, & k = 0 \\ -\log\left(PDF_j(k)\right), & k \neq 0. \end{cases}$$
  
  最终的损失函数表示为：
  $$L_{\text{SD}} = \sum_j P_j(k) \left(1 - \frac{2 \sum (S \cdot G) + \epsilon}{\sum S + \sum G + \epsilon} \right),$$
  其中 $S$ 和 $G$ 分别表示预测和目标分割结果。

- 双向连通性损失 $L_{\text{Bicon}}$：
  $L_{\text{Bicon}}$ 包括两部分，分别为方向对齐损失和连通性匹配损失，用于建模像素间的连通关系。

##### 1.2 存在的不足
1. **对低频结构的不足**：
$L_{\text{Bicon}}$ 主要关注局部连通性，忽视了整体拓扑结构和低频分量的保留。

2. **拓扑一致性不足**：
$L_{\text{SDL}}$ 的标签加权策略主要基于像素分布，未能直接建模拓扑一致性。

3. **缺乏多尺度特性**：
原始损失函数未考虑多尺度特征提取和频域表示，在引入多尺度特征后，损失函数需要进行扩展。

#### 2. 扩展改进后的损失函数

##### 2.1 改进设计

改进的损失函数包括三个主要部分：连通性损失、多尺度频域损失和拓扑保留损失。总损失函数定义为：
$$L_{\text{total}} = w_c L_{\text{connect}} + w_f L_{\text{freq}} + w_t L_{\text{topo}},$$
其中 $w_c, w_f, w_t$ 分别是权重系数，控制不同损失项的贡献。

- 连通性损失 $L_{\text{connect}}$：
  在原始双向连通性损失的基础上，进一步优化了边缘约束和一致性权重，以提升对复杂连通性的捕获能力。

- 多尺度频域损失 $L_{\text{freq}}$：
  通过快速傅里叶变换 (FFT) 将预测值和目标值转换到频域，并计算对数幅度谱的均方误差 (MSE)：
  $$L_{\text{freq}} = \frac{1}{3} \sum_{s \in {1, 2, 4}} \text{MSE} \left(\log(|FFT_s(P)|), \log(|FFT_s(T)|)\right),$$
  其中 $P$ 和 $T$ 分别表示预测值和目标值，$s$ 表示多尺度下的降采样因子。

- 拓扑保留损失 $L_{\text{topo}}$：
使用方向梯度的一致性和边缘增强策略建模拓扑结构：
  - 梯度方向一致性：
    $$L_{\text{dir}} = \left(1 - \cos(\theta_P - \theta_T)\right) \cdot W_{\text{edge}},$$
    其中 $\theta$ 表示梯度方向，$W_{\text{edge}}$ 为边缘加权。

  - 边缘增强损失：
    $$L_{\text{edge}} = \text{MAE}(M_P \cdot M_T, M_T^2),$$
    其中 $M$ 为梯度幅值。

2.2 改进点分析
1. **多尺度建模**：
$L_{\text{freq}}$ 在不同尺度下提取频域特征，显著增强了模型对全局和局部结构的捕获能力。

2. **拓扑保留**：
$L_{\text{topo}}$ 通过梯度方向一致性和边缘增强有效避免了拓扑断裂问题。

3. **计算效率**：
改进后的损失函数保持了计算复杂度的低增长，同时提升了性能。

改进的损失函数通过整合连通性、多尺度频域表示和拓扑保留，在医学图像分割任务中取得了更优的解剖一致性和全局连通性表现。


## 环境要求
详细的环境配置请查看[requirements.txt](requirements.txt)。

## 代码结构
本仓库的主要结构和重要文件或函数如下：
```
  - train.py: 主文件，定义参数和GPU选择等
  - solver.py: 训练和测试的详细实现
  losses: 增强型损失函数实现
  data_loader: 数据加载文件和SDL权重
  model: 
    - DconnNet.py: 基础模型实现
    - attention.py: 解码器注意力模块
    - mrde.py: 多分辨率方向增强
    - glfi.py: 全局-局部特征集成
  scripts: 不同数据集的训练脚本
```

### 数据集与训练
#### 支持的数据集
1. **Retouch**
```
/retouch
  /Cirrus ### 设备类型，同样适用于Spectrailis和Topcon
    /train
      /TRAIN002 ### 数据卷ID
        /mask ### 存放.png格式的掩码
        /orig ### 存放.png格式的原始图像
```

2. **ISIC2018**
调整大小后的数据集参考[此处](https://github.com/duweidai/Ms-RED)，主要超参数如下：
```
/ISIC2018_npy_all_224_320
  /image
  /label

图像尺寸: (224, 320)
批次大小: 10
训练轮数: 200
初始学习率: 1e-4
优化器: Adam (权重衰减 1e-8)
学习率调度: CosineAnnealingWarmRestarts (T_0=15, T_mult=2, eta_min = 0.00001)
```

3. **CHASEDB1**
```
/CHASEDB1
  /img
  /gt
```

#### 训练说明
1. **使用提供的数据集**：
   - 按上述方式组织数据
   - 使用`scripts/`中的对应脚本
   - 启用增强模块进行训练：
     ```bash
     python train.py --use_attention --use_mrde --use_glfi
     ```

2. **使用自定义数据集**：
   - 准备数据加载器
   - 在`train.py`中配置网络设置
   - 指定要使用的增强模块
   - 运行训练脚本

### 实现细节
**重要：请遵循以下步骤确保正确实现**
1. 基础模型设置：
   - 从`/model`获取模型文件
   - 使用`losses/connect_loss.py`中的增强损失函数
   - 根据任务选择适当的前向传播：
     * 单类别：`connect_loss.single_class_forward`
     * 多类别：`connect_loss.multi_class_forward`

2. 测试阶段：
   - 遵循`/solver.py`中的官方程序
   - 单类别：`sigmoid --> threshold --> Bilateral_voting`
   - 多类别：`Bilateral_voting --> topK (softmax + topK)`
   - 配置`hori_translation`和`verti_translation`用于矩阵平移

### 补充说明
- SDL损失使用说明：预先计算掩码尺寸分布(形状: C×N)并保存为.npy文件
- 请仔细检查数据维度，特别是在`loss/connect_loss.py`中
