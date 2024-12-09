import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalPyramidPooling(nn.Module):
    """Directional Pyramid Pooling module for multi-scale directional feature extraction"""
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.levels = [1, 2, 4]  # Pyramid levels
        reduced_channels = in_channels // reduction_ratio
        
        # 添加输入归一化
        self.input_norm = nn.GroupNorm(8, in_channels)
        
        # 修改卷积层，添加更严格的归一化
        self.dir_convs = nn.ModuleList([
            nn.Sequential(
                # 使用GroupNorm替代BatchNorm，更稳定
                nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
                nn.GroupNorm(4, reduced_channels),
                nn.ReLU(inplace=False),
                # 使用深度可分离卷积减少参数量
                nn.Conv2d(reduced_channels, reduced_channels, 3, 
                         padding=1, groups=reduced_channels, bias=False),
                nn.GroupNorm(4, reduced_channels),
                nn.ReLU(inplace=False),
                # 点卷积
                nn.Conv2d(reduced_channels, reduced_channels*4, 1, bias=False),  # 减少通道数
                nn.GroupNorm(4, reduced_channels*4),
                nn.ReLU(inplace=False)
            ) for _ in self.levels
        ])
        
        # 添加特征缩放因子
        self.scale_factors = nn.Parameter(torch.ones(len(self.levels)) * 0.1)
        
        # 修改fusion层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + reduced_channels*4*len(self.levels), 
                     in_channels, 1, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=False),
            nn.Dropout2d(0.1)  # 添加dropout
        )
        
    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            out: Enhanced feature map with multi-scale directional information
        """
        # 输入归一化
        x = self.input_norm(x)
        h, w = x.shape[2:]
        outputs = [x]  # Keep the input feature
        
        # Process each pyramid level
        for i, level in enumerate(self.levels):
            # 确保池化后的尺寸至少为1
            kernel_h = max(1, h//level)
            kernel_w = max(1, w//level)
            feat = F.adaptive_avg_pool2d(x, (kernel_h, kernel_w))
            
            # Apply directional convolutions
            feat = self.dir_convs[i](feat)
            
            # Upsample back to original size
            feat = F.interpolate(feat, (h,w), 
                               mode='bilinear', align_corners=False)
            
            # 应用可学习的缩放因子
            feat = feat * self.scale_factors[i]
            
            # 添加L2正则化
            feat = feat / (torch.norm(feat, p=2, dim=1, keepdim=True) + 1e-6)
            
            outputs.append(feat)
            
        # Concatenate all features
        out = torch.cat(outputs, dim=1)
        
        # Final fusion
        out = self.fusion(out)
        
        # 残差连接前的特征归一化
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-6)
        return out

class MRDE(nn.Module):
    """Multi-Resolution Directional Enhancement Module"""
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        
        self.dpp = DirectionalPyramidPooling(channels, reduction_ratio)
        
        # 修改channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction_ratio, 1, bias=False),
            nn.GroupNorm(4, channels//reduction_ratio),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels//reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 修改spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels//2, 1, bias=False),
            nn.GroupNorm(4, channels//2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels//2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 添加输出缩放因子
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用更保守的初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 输入特征归一化
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-6)
        
        # 方向性特征增强
        feat = self.dpp(x)
        
        # 应用attention
        channel_att = self.channel_att(feat)
        spatial_att = self.spatial_att(feat)
        
        # 特征融合
        feat = feat * channel_att * spatial_att
        
        # 残差连接
        out = x + feat * self.output_scale
        
        # 输出归一化
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-6)
        return out