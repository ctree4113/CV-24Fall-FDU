# Enhanced Directional Connectivity Medical Image Segmentation Model

[English](README.md) | [中文](README_CN.md)

This repository contains our enhanced version of DconnNet for medical image segmentation tasks. This work is based on the CVPR2023 paper "Medical Image Segmentation via Directional Connectivity" [[Paper](https://arxiv.org/pdf/2304.00145.pdf)].

## Main Innovations and Improvements Analysis

Our enhanced version comprehensively improves upon the original DconnNet, particularly suited for medical images with complex anatomical structures. Our improvements include:

### Improved Decoder Attention Mechanism

---

In the decoding phase, we dynamically weight and fuse feature maps from different levels, enhancing feature interaction and segmentation detail performance.

---

#### Improvement Method

1. **Dynamic Feature Fusion:**  
   Concatenate features of different resolutions ($c_1, c_2, c_3, c_4$) from the decoder path to form joint features:
   $$
   F_{\text{concat}} = \text{Concat}(c_1, c_2, c_3, c_4).
   $$

   Use $1 \times 1$ convolution and $3 \times 3$ convolution to extract global and local context information, generating dynamic weight feature maps:
   $$
   F_{\text{attention}} = \text{Conv}_{1 \times 1}(\text{ReLU}(\text{Conv}_{3 \times 3}(F_{\text{concat}}))).
   $$

2. **Weight-based Fusion:**  
   Dynamic weighting mechanism fuses features from each layer to generate final decoder input:
   $$
   F_{\text{fused}} = \sum_{i=1}^4 \alpha_i \cdot c_i, \quad \alpha_i \text{ determined by } F_{\text{attention}}.
   $$

---

#### Analysis of Improvements

1. **Enhanced Cross-level Feature Interaction:**  
   Dynamic weighting mechanism makes feature fusion more selective, helping capture multi-resolution contextual information.
   
2. **Lightweight Design:**  
   Module uses simple convolutions, low computational complexity, easy to integrate, and doesn't increase model complexity.

---

Through these improvements, the model achieves dynamic weighting fusion of features at different resolutions while maintaining computational efficiency, effectively enhancing segmentation consistency and accuracy.

### Replacing Original SDE Module with MRDE Module

---

The MRDE module, through multi-resolution architecture, lightweight convolutions, and more stable directional enhancement strategies, not only significantly outperforms SDE in computational efficiency but also better maintains gradient stability and directional information extraction capabilities.

---

#### 1. Original SDE Module

##### 1.1 Module Structure
The SDE module design includes three key parts: **Directional Prior Extraction**, **Channel-wise Slicing**, and **Sub-path Excitation**. The process is as follows:

1. **Directional Prior Extraction**  
   SDE uses direction-embedded global feature $e_5$ to generate directional prior $\alpha_{\text{prior}}$. The specific process is:
   - $e_5$ is upsampled to input size, producing directional output $X_{\text{prior}}$:
     $$
     X_{\text{prior}} = \text{Conv}(\text{Upsample}(e_5)).
     $$
   - Using Global Average Pooling (GAP) and $1 \times 1$ convolution layer to extract directional embedding $v_{\text{prior}}$:
     $$
     v_{\text{prior}} = \delta(W_1 \cdot \text{GAP}(X_{\text{prior}})),
     $$
     where $\delta$ is ReLU activation function.
   - Directional prior $\alpha_{\text{prior}}$ is normalized through a Sigmoid function:
     $$
     \alpha_{\text{prior}} = \sigma(W_2 \cdot v_{\text{prior}}).
     $$

2. **Channel-wise Slicing**  
   $e_5$ and $\alpha_{\text{prior}}$ are sliced along channel dimension into $i$ groups of sub-channels $e_5^i$ and $\alpha_{\text{prior}}^i$, each group responsible for modeling features in different directions.

3. **Sub-path Excitation (SPE)**  
   Each sub-channel group $e_5^i$ goes through dual attention modules (PAM and CAM) to extract contextual correlations and channel dependencies:
   $$
   e_5^{i'} = \text{PAM}(\text{CAM}(e_5^i)).
   $$
   Using directional prior $\alpha_{\text{prior}}^i$ for selective activation of features:
   $$
   e_{\text{SDE}}^i = W_3^i (\alpha_{\text{prior}}^i \cdot e_5^{i'}) + e_5^i.
   $$
   Finally, outputs from each sub-path are stacked and re-encoded as new feature $e_{\text{SDE}}$.

---

##### 1.2 Limitations
1. **Single Resolution Feature Extraction:**  
   SDE module extracts directional features only at single resolution $e_5$, making it difficult to capture both global topology and local details simultaneously.

2. **High Computational Complexity:**  
   Each sub-path uses independent PAM and CAM, leading to high computational and memory overhead.

3. **Gradient Instability:**  
   In multi-path parallel computation of SDE module, long gradient propagation paths easily cause gradient explosion or vanishing problems.

---

#### 2. Improved MRDE Module

##### 2.1 Module Structure
The MRDE module design focuses on multi-resolution enhancement, divided into the following key parts:

1. **Multi-resolution Pyramid Feature Extraction:**  
   Input feature $X$ is downsampled to multiple resolutions $\{1, 2, 4\}$, generating features $X_s$ at different scales:
   $$
   X_s = \text{AdaptiveAvgPool}(X, (H/s, W/s)), \quad s \in \{1, 2, 4\}.
   $$

2. **Directional Convolution Enhancement (Directional ConvBlock):**  
   Each scale feature $X_s$ is processed through directional convolution module to extract directional information:
   $$
   X_s' = \text{DepthwiseConv}(X_s) + \text{PointwiseConv}(X_s).
   $$

3. **Multi-resolution Feature Fusion:**  
   All scale directional enhanced features $\{X_1', X_2', X_4'\}$ are upsampled to uniform size and concatenated:
   $$
   X' = \text{Concat}([X_1', X_2', X_4'], \text{dim}=C).
   $$

4. **Attention Mechanism Optimization:**  
   - **Channel Attention:** Generate channel weights $\alpha_{\text{channel}}$ through global average pooling and MLP:
     $$
     \alpha_{\text{channel}} = \sigma(\text{MLP}(\text{GAP}(X'))).
     $$
   - **Spatial Attention:** Generate spatial weights $\alpha_{\text{spatial}}$ through $7 \times 7$ convolution:
     $$
     \alpha_{\text{spatial}} = \sigma(\text{Conv}_{7 \times 7}(X')).
     $$
   Final enhanced feature:
   $$
   X_{\text{MRDE}} = X' \cdot \alpha_{\text{channel}} \cdot \alpha_{\text{spatial}}.
   $$

5. **Normalization and Residual Connection:**  
   Output features maintain stability through L2 normalization and residual path:
   $$
   X_{\text{output}} = \frac{X_{\text{MRDE}}}{\|X_{\text{MRDE}}\|_2 + \epsilon}.
   $$

---

##### 2.2 Analysis of Improvements
1. **Multi-resolution Feature Capture:**  
   MRDE module can extract global and local features simultaneously at multiple scales, more suitable for capturing complex structures compared to SDE's single-scale processing.

2. **Lightweight Design:**  
   MRDE uses depth-separable convolutions instead of PAM and CAM in SDE, significantly reducing computational complexity.

  - **SDE Module:**  
    For input size $C \times H \times W$, SDE's complexity is:
    $$
    \mathcal{O}_{\text{SDE}} = 8 \cdot (C^2HW + CHW) \approx 8C^2HW.
    $$
  - **MRDE Module:**  
    MRDE's multi-resolution processing complexity is:
    $$
    \mathcal{O}_{\text{MRDE}} = \sum_{s \in \{1, 2, 4\}} \left( C^2H^2/s^2 \right) \approx 1.75C^2HW.
    $$

3. **Gradient Stability:**  
   MRDE calculates directional features independently at each scale, reducing gradient propagation paths, combined with input-output normalization to avoid gradient explosion.
  - **SDE Module:**  
    Independent gradient calculation between sub-paths leads to higher instability:
    $$
    \nabla L_{e_5^i} = \nabla L_{\text{concat}} \cdot \frac{\partial \text{PAM}(\text{CAM}(e_5^i))}{\partial e_5^i}.
    $$
  - **MRDE Module:**  
    MRDE's normalization and directional convolution reduce gradient fluctuation.

---

The MRDE module, building upon SDE, significantly improves the accuracy and efficiency of directional feature extraction through multi-resolution directional enhancement design, featuring better gradient stability and robustness. This makes the MRDE module demonstrate superior topological consistency and computational efficiency in medical image segmentation tasks, particularly suitable for complex structures and high-resolution inputs.

### Using GLFI Module to Enhance Global and Local Feature Interaction

---

The GLFI module achieves significant improvements in capturing complex structures and optimizing boundary details through combining multi-scale global feature extraction, edge enhancement, and graph attention mechanism, providing a more robust and efficient solution in global-local feature interaction. This module is integrated into DconnNet's decoder path as a key step for feature refinement and fusion.

---

#### 1. Module Structure Design

The GLFI (Global-Local Feature Interaction) module aims to enhance global and local feature interaction through multi-perspective feature modeling. Its design includes the following core parts:

1. **Multi-scale Global Feature Extraction**  
   To capture global context information under different receptive fields, the GLFI module extracts global features through four dilated convolution branches:
   $$
   F_{\text{global}} = \text{Concat}\left[\text{DilatedConv}_{r=1}, \text{DilatedConv}_{r=2}, \text{DilatedConv}_{r=4}, \text{DilatedConv}_{r=8}\right],
   $$
   where $r$ is the dilation rate of dilated convolution. The output features from each branch represent global information at specific scales and are concatenated along the channel dimension to form multi-scale global feature representation.

2. **Edge Detection Enhancement**  
   Use enhanced Sobel operator to extract edge features and refine these edge information through learnable convolution module:
   $$
   F_{\text{edge}} = \text{Refine}(\text{Sobel}_x(F) \oplus \text{Sobel}_y(F)),
   $$
   where $\text{Sobel}_x$ and $\text{Sobel}_y$ are Sobel filters in X and Y directions respectively, $\oplus$ represents feature concatenation, $\text{Refine}$ is the convolution network for refining edge features. Through this way, GLFI can enhance the model's perception of boundaries and local details.

3. **Graph Attention Feature Refinement**  
   In the interaction process between global and local features, use multi-head graph attention mechanism to model global relationships between features. Specific steps include:
   - Linear transformation generates query ($Q$), key ($K$), value ($V$):
     $$
     Q = W_q F, \quad K = W_k F, \quad V = W_v F,
     $$
     where $W_q, W_k, W_v$ are linear transformation matrices.
   - Calculate attention weights:
     $$
     A = \text{Softmax}\left(\frac{Q \cdot K^\top}{\sqrt{d_k}}\right),
     $$
     where $d_k$ is scaling factor used to stabilize gradients.
   - Generate attention features based on attention weights and feature values $V$:
     $$
     F_{\text{attn}} = A \cdot V.
     $$
   - Project output:
     $$
     F_{\text{out}} = W_o F_{\text{attn}},
     $$
     where $W_o$ is output projection matrix.

4. **Feature Fusion and Optimization**  
   Fuse global features $F_{\text{global}}$, edge features $F_{\text{edge}}$ and graph attention features $F_{\text{attn}}$, optimize feature interaction through channel and spatial attention:
   $$
   F_{\text{fusion}} = \alpha_{\text{channel}} \cdot F_{\text{global}} + \alpha_{\text{spatial}} \cdot F_{\text{attn}},
   $$
   where $\alpha_{\text{channel}}, \alpha_{\text{spatial}}$ are attention weights generated through global average pooling and convolution respectively, ensuring effectiveness of feature fusion.

---

#### 2. Application in Model

The GLFI module is integrated into DconnNet's decoder path, mainly acting on feature refinement and fusion of intermediate layers. The specific application process is as follows:
1. **Multi-scale Feature Post-processing:**  
  In each layer of DconnNet's decoder path ($d_4, d_3, d_2, d_1$), refine context features through GLFI module, decoder feature $d_k$ at layer $k$ will be input to GLFI module to enhance local boundaries and global coherence.

2. **Multi-resolution Feature Fusion:**  
  Corresponding skip connection features from encoder path (multi-resolution features generated through ResNet encoder, i.e., $c_4, c_3, c_2, c_1$) will be fused with decoder features processed by GLFI, combining global semantic information from encoder path with local reconstruction information from decoder path to generate more refined segmentation feature representation. Multi-resolution features are further optimized at each decoder level, ensuring the model can capture both details and global topological structure.

3. **Final Segmentation Optimization:**  
  After completing fusion and refinement of features at each decoder layer, these features processed by GLFI module will be uniformly input to final decoder module. The final decoder module adopts a lightweight design, receiving fused features from each level (such as $d_4^{\text{GLFI}}, d_3^{\text{GLFI}}, d_2^{\text{GLFI}}, d_1^{\text{GLFI}}$), generating full-resolution segmentation results through progressive upsampling and feature fusion operations. The final segmentation result combines advantages of features from all levels, able to perform better on fine structures (like blood vessels) and global semantic consistency (like organ contours). Through this hierarchical feature fusion and optimization, the model effectively improves segmentation accuracy and robustness while ensuring processing capability for complex boundaries and multi-scale structures.

---

#### 3. Analysis of Improvements

1. **Multi-scale Modeling of Global Features**  
   Through dilated convolutions with different dilation rates, GLFI module can capture rich multi-scale global context information, more suitable for segmentation tasks with complex structures and long-range dependencies.

2. **Enhanced Edge Detection Capability**  
   Using Sobel operator and learnable convolution to refine edge information can significantly improve model's sensitivity to complex boundaries and small targets, especially performing well in medical images.

3. **Global Dependency Modeling with Graph Attention Mechanism**  
   Compared to traditional channel or spatial attention mechanisms, graph attention mechanism can more effectively model complex dependencies between features globally, thus enhancing semantic consistency and topological structure preservation.

4. **Lightweight Design with Fusion Optimization**  
   While ensuring feature expression capability, reduce redundant computation through attention mechanism, maintaining good efficiency even with high-resolution inputs.

---

The GLFI module provides stronger semantic understanding and boundary refinement capabilities for medical image segmentation tasks through integrating global and local feature interaction design. Through collaborative optimization of multi-scale global feature extraction, edge enhancement, and graph attention mechanism, GLFI significantly improves topological consistency and global coherence of segmentation results, providing an effective solution for segmentation tasks in high-resolution and complex structure scenarios.

### Extending and Improving Loss Function Based on Original Loss

The improved loss function enhances model robustness to imbalanced data and significantly improves anatomical structure coherence and directional consistency through introducing multi-scale frequency domain loss, topology preservation loss, and distributed weight design.

#### 1. Original Loss Function

##### 1.1 Analysis of Original Loss Function Structure

The original total loss function is defined as:
$$
L_{\text{total}} = L_{\text{main}} + 0.3 \cdot L_{\text{prior}},
$$
where $L_{\text{main}}$ and $L_{\text{prior}}$ correspond to model's main output and auxiliary output from SDE module respectively. These two parts each consist of Size Density Loss (SDL) and Bidirectional Connectivity Loss (Bicon Loss):
$$
L = L_{\text{SD}} + L_{\text{Bicon}}.
$$

- Size Density Loss $L_{\text{SDL}}$:
  SDL introduces a weighting mechanism based on label size distribution for medical data imbalance problem. First, calculate probability density function $PDF_j(k)$ of label size distribution for each class in all training data, then calculate corresponding weight coefficient $P_j(k)$ for label size $k$ of each sample:
  $$
  P_j(k) =
  \begin{cases}
  1, & k = 0 \\
  -\log\left(PDF_j(k)\right), & k \neq 0.
  \end{cases}
  $$
  The final loss function is expressed as:
  $$
  L_{\text{SD}} = \sum_j P_j(k) \left(1 - \frac{2 \sum (S \cdot G) + \epsilon}{\sum S + \sum G + \epsilon} \right),
  $$
  where $S$ and $G$ represent predicted and target segmentation results respectively.

- Bidirectional Connectivity Loss $L_{\text{Bicon}}$:
  $L_{\text{Bicon}}$ includes two parts, directional alignment loss and connectivity matching loss, used to model connectivity between pixels.
  $$
  L_{\text{Bicon}} = L_{\text{decouple}} + L_{\text{con\_const}}.
  $$

##### 1.2 Limitations
1. **Insufficient for Low-frequency Structures**:
$L_{\text{Bicon}}$ mainly focuses on local connectivity, ignoring overall topological structure and low-frequency component preservation.

2. **Insufficient Topological Consistency**:
$L_{\text{SDL}}$'s label weighting strategy is mainly based on pixel distribution, not directly modeling topological consistency.

3. **Lack of Multi-scale Characteristics**:
Original loss function doesn't consider multi-scale feature extraction and frequency domain representation, loss function needs to be extended after introducing multi-scale features.

#### 2. Extended Improved Loss Function

##### 2.1 Improvement Design

The improved loss function includes three main parts: connectivity loss, multi-scale frequency domain loss, and topology preservation loss. Total loss function is defined as:
$$
L_{\text{total}} = w_c L_{\text{connect}} + w_f L_{\text{freq}} + w_t L_{\text{topo}},
$$
where $w_c, w_f, w_t$ are weight coefficients controlling contribution of different loss terms.

- Connectivity Loss $L_{\text{connect}}$:
  Further optimizes edge constraints and consistency weights based on original bidirectional connectivity loss to enhance capability of capturing complex connectivity.

- Multi-scale Frequency Domain Loss $L_{\text{freq}}$:
  Convert predicted values and target values to frequency domain through Fast Fourier Transform (FFT) and calculate Mean Square Error (MSE) of logarithmic magnitude spectrum:
  $$
  L_{\text{freq}} = \frac{1}{3} \sum_{s \in {1, 2, 4}} \text{MSE} \left(\log(|FFT_s(P)|), \log(|FFT_s(T)|)\right),
  $$
  where $P$ and $T$ represent predicted and target values respectively, $s$ represents downsampling factor under multi-scale.

- Topology Preservation Loss $L_{\text{topo}}$:
Use directional gradient consistency and edge enhancement strategy to model topological structure:
  - Gradient Direction Consistency:
    $$
    L_{\text{dir}} = \left(1 - \cos(\theta_P - \theta_T)\right) \cdot W_{\text{edge}},
    $$
    where $\theta$ represents gradient direction, $W_{\text{edge}}$ is edge weighting.

  - Edge Enhancement Loss:
    $$
    L_{\text{edge}} = \text{MAE}(M_P \cdot M_T, M_T^2),
    $$
    where $M$ is gradient magnitude.

2.2 Analysis of Improvements
1. **Multi-scale Modeling**:
$L_{\text{freq}}$ extracts frequency domain features at different scales, significantly enhancing model's capability to capture global and local structures.

2. **Topology Preservation**:
$L_{\text{topo}}$ effectively avoids topological breaking problems through gradient direction consistency and edge enhancement.

3. **Computational Efficiency**:
Improved loss function maintains low growth in computational complexity while improving performance.

The improved loss function achieves better anatomical consistency and global connectivity performance in medical image segmentation tasks through integrating connectivity, multi-scale frequency domain representation, and topology preservation.


## Environment Requirements
For detailed environment configuration, please check [requirements.txt](requirements.txt).

## Code Structure
The main structure and important files or functions of this repository are as follows:
```
  - train.py: Main file, defines parameters and GPU selection etc.
  - solver.py: Detailed implementation of training and testing
  losses: Enhanced loss function implementation
  data_loader: Data loading files and SDL weights
  model: 
    - DconnNet.py: Base model implementation
    - attention.py: Decoder attention module
    - mrde.py: Multi-resolution directional enhancement
    - glfi.py: Global-local feature integration
  scripts: Training scripts for different datasets
```

### Dataset and Training
#### Supported Datasets
1. **Retouch**
```
/retouch
  /Cirrus ### Device type, also applies to Spectrailis and Topcon
    /train
      /TRAIN002 ### Data volume ID
        /mask ### Store masks in .png format
        /orig ### Store original images in .png format
```

2. **ISIC2018**
For resized dataset refer to [here](https://github.com/duweidai/Ms-RED), main hyperparameters are as follows:
```
/ISIC2018_npy_all_224_320
  /image
  /label

Image size: (224, 320)
Batch size: 10
Training epochs: 200
Initial learning rate: 1e-4
Optimizer: Adam (weight decay 1e-8)
Learning rate schedule: CosineAnnealingWarmRestarts (T_0=15, T_mult=2, eta_min = 0.00001)
```

3. **CHASEDB1**
```
/CHASEDB1
  /img
  /gt
```

#### Training Instructions
1. **Using Provided Datasets**:
   - Organize data as described above
   - Use corresponding script from `scripts/`
   - Enable enhancement modules for training:
     ```bash
     python train.py --use_attention --use_mrde --use_glfi
     ```

2. **Using Custom Datasets**:
   - Prepare data loader
   - Configure network settings in `train.py`
   - Specify enhancement modules to use
   - Run training script

### Implementation Details
**Important: Please follow these steps for correct implementation**
1. Base Model Setup:
   - Get model files from `/model`
   - Use enhanced loss function from `losses/connect_loss.py`
   - Choose appropriate forward pass based on task:
     * Single class: `connect_loss.single_class_forward`
     * Multi class: `connect_loss.multi_class_forward`

2. Testing Phase:
   - Follow official procedure in `/solver.py`
   - Single class: `sigmoid --> threshold --> Bilateral_voting`
   - Multi class: `Bilateral_voting --> topK (softmax + topK)`
   - Configure `hori_translation` and `verti_translation` for matrix translation

### Additional Notes
- SDL loss usage: Pre-calculate mask size distribution (shape: C×N) and save as .npy file
- Carefully check data dimensions, especially in `loss/connect_loss.py`
