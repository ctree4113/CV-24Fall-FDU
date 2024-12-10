import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalConvBlock(nn.Module):
    """Directional Convolution Block with gradient stability"""
    def __init__(self, channels):
        super().__init__()
        
        # Depthwise separable convolution for directional feature extraction
        self.dir_conv = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=False),
            # Pointwise conv
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels)
        )
        
        # Learnable scale factor
        self.scale = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.1
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input normalization
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-6)
        
        # Directional feature extraction
        feat = self.dir_conv(x)
        
        # Scale and residual connection
        scale = torch.sigmoid(self.scale) * 0.1
        out = x + feat * scale
        
        # Output normalization
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-6)
        
        return out

class MRDE(nn.Module):
    """Multi-Resolution Directional Enhancement Module with gradient stability"""
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        
        # Multi-scale directional feature extraction
        self.scales = [1, 2, 4]  # Remove scale 8 to prevent too small feature maps
        reduced_channels = channels // reduction_ratio
        
        # Multi-scale branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                # Channel reduction
                nn.Conv2d(channels, reduced_channels, 1, bias=False),
                nn.GroupNorm(4, reduced_channels),
                nn.ReLU(inplace=False),
                # Directional feature extraction
                DirectionalConvBlock(reduced_channels),
                # Channel expansion
                nn.Conv2d(reduced_channels, channels, 1, bias=False),
                nn.GroupNorm(8, channels)
            ) for _ in self.scales
        ])
        
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction_ratio, 1, bias=False),
            nn.GroupNorm(4, channels//reduction_ratio),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels//reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels//2, 1, bias=False),
            nn.GroupNorm(4, channels//2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels//2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(self.scales), channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=False)
        )
        
        # Learnable scale factor
        self.scale = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.1
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input normalization
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-6)
        
        # Multi-scale feature extraction
        feats = []
        for i, branch in enumerate(self.branches):
            # Adaptive pooling for different scales
            scale = self.scales[i]
            if scale > 1:
                h, w = x.shape[2:]
                h_s, w_s = h // scale, w // scale
                # Ensure minimum size
                h_s, w_s = max(1, h_s), max(1, w_s)
                x_scaled = F.adaptive_avg_pool2d(x, (h_s, w_s))
            else:
                x_scaled = x
            
            # Process features
            feat = branch(x_scaled)
            
            # Upsample if needed
            if scale > 1:
                feat = F.interpolate(feat, size=x.shape[2:], 
                                   mode='bilinear', align_corners=False)
            
            feats.append(feat)
        
        # Feature fusion
        feat = torch.cat(feats, dim=1)
        feat = self.fusion(feat)
        
        # Apply attention
        channel_weight = self.channel_att(feat)
        spatial_weight = self.spatial_att(feat)
        feat = feat * channel_weight * spatial_weight
        
        # Scale and residual connection
        scale = torch.sigmoid(self.scale) * 0.1
        out = x + feat * scale
        
        # Output normalization
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-6)
        
        return out