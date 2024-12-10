import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for feature refinement"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Node feature transformation
        self.node_transform = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=False)
        )
        
        # Multi-head attention (4 heads)
        self.num_heads = 4
        head_dim = channels // self.num_heads
        
        self.q_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels)
        )
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
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
        B, C, H, W = x.shape
        
        # Transform node features
        x = self.node_transform(x)
        
        # Multi-head attention
        q = self.q_conv(x).view(B, self.num_heads, C // self.num_heads, -1)
        k = self.k_conv(x).view(B, self.num_heads, C // self.num_heads, -1)
        v = self.v_conv(x).view(B, self.num_heads, C // self.num_heads, -1)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention and reshape
        out = torch.matmul(attn, v)
        out = out.view(B, C, H, W)
        
        # Output projection with residual
        out = self.proj(out)
        
        # Feature normalization
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-6)
        
        return out

class GLFI(nn.Module):
    """Global-Local Feature Interaction Module with enhanced graph structure"""
    def __init__(self, channels):
        super().__init__()
        
        # Global feature extraction with dilated convolutions
        self.global_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels//4, 3, padding=rate, dilation=rate, 
                         groups=channels//4, bias=False),
                nn.GroupNorm(4, channels//4),
                nn.ReLU(inplace=False)
            ) for rate in [1, 2, 4, 8]
        ])
        
        # Edge detection with learnable Sobel filters
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels//2, 1, bias=False),
            nn.GroupNorm(4, channels//2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels//2, channels, 3, padding=1, groups=channels//2, bias=False),
            nn.GroupNorm(8, channels)
        )
        
        # Graph attention for feature refinement
        self.graph_attention = nn.Sequential(
            GraphAttentionLayer(channels),
            nn.ReLU(inplace=False),
            GraphAttentionLayer(channels)
        )
        
        # Feature fusion with channel and spatial attention
        self.fusion = nn.Sequential(
            # Channel attention
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels*2, channels//4, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels//4, channels*2, 1, bias=False),
            nn.Sigmoid(),
            # Spatial attention
            nn.Conv2d(channels*2, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(8, channels)
        )
        
        # Learnable scale factor
        self.scale = nn.Parameter(torch.zeros(1))
        
        # Register Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32).view(1, 1, 3, 3) * 0.1
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32).view(1, 1, 3, 3) * 0.1
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.1
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)
    
    def get_edge_features(self, x):
        """Extract edge features with enhanced gradient stability"""
        B, C, H, W = x.shape
        
        # Input normalization
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-6)
        
        # Apply Sobel operators
        x_pad = F.pad(x.view(B*C, 1, H, W), (1, 1, 1, 1), mode='reflect')
        edge_x = F.conv2d(x_pad, self.sobel_x)
        edge_y = F.conv2d(x_pad, self.sobel_y)
        
        # Compute edge magnitude
        edge = torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-6).view(B, C, H, W)
        
        # Edge feature normalization
        edge = edge / (torch.norm(edge, p=2, dim=1, keepdim=True) + 1e-6)
        
        return torch.cat([x, edge], dim=1)
    
    def forward(self, x):
        # Input normalization
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-6)
        
        # Global feature extraction with dilated convolutions
        global_feats = []
        for branch in self.global_branch:
            global_feats.append(branch(x))
        global_feat = torch.cat(global_feats, dim=1)
        
        # Edge feature extraction
        edge_feat = self.get_edge_features(x)
        edge_feat = self.edge_conv(edge_feat)
        
        # Graph-based feature refinement
        refined_feat = self.graph_attention(edge_feat)
        
        # Feature fusion
        feat = torch.cat([global_feat, refined_feat], dim=1)
        feat = self.fusion(feat)
        
        # Scale and residual connection
        scale = torch.sigmoid(self.scale) * 0.1
        out = x + feat * scale
        
        # Output normalization
        out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-6)
        
        return out