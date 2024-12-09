import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvBlock(nn.Module):
    """Graph Convolution Block for feature interaction"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Node feature transform
        self.node_transform = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Edge attention
        self.edge_att = nn.Sequential(
            nn.Conv2d(channels*2, channels//2, 1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.channels, f"Input channels {c} doesn't match expected channels {self.channels}"
        
        # Transform node features
        nodes = self.node_transform(x)  # B,C,H,W
        nodes_flat = nodes.view(b, c, -1)  # B,C,N where N = H*W
        
        # Compute edge attention weights
        nodes_1 = nodes_flat.permute(0,2,1).contiguous()  # B,N,C
        nodes_2 = nodes_flat  # B,C,N
        
        N = nodes_1.size(1)
        edge_feat = torch.cat([
            nodes_1.unsqueeze(2).expand(-1,-1,N,-1),      # B,N,N,C
            nodes_2.permute(0,2,1).unsqueeze(1).expand(-1,N,-1,-1)  # B,N,N,C
        ], dim=-1)  # B,N,N,2C
        
        adj = self.edge_att(edge_feat.permute(0,3,1,2))  # B,1,N,N
        adj = F.softmax(adj, dim=-1)
        
        # Graph convolution
        out = torch.matmul(nodes_flat, adj.squeeze(1))  # B,C,N
        out = out.view(b,c,h,w)
        
        return out

class GLFI(nn.Module):
    """Global-Local Feature Interaction Module"""
    def __init__(self, channels, num_blocks=3):
        super().__init__()
        
        # Global feature processing
        self.global_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False)
        )
        
        # Edge feature processing
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False)
        )
        
        # Attention fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        # Add normalization layers
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.layer_norm = nn.LayerNorm([channels, 1, 1])
        
        # Register Sobel filters as buffers
        self.register_buffer('sobel_x', 
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                        dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                        dtype=torch.float32).view(1, 1, 3, 3))
        
    def get_edge_features(self, x):
        # Apply instance normalization
        x = self.instance_norm(x)
        
        b, c, h, w = x.shape
        x_pad = F.pad(x.view(b*c, 1, h, w), (1, 1, 1, 1), mode='reflect')
        
        edge_x = F.conv2d(x_pad, self.sobel_x)
        edge_y = F.conv2d(x_pad, self.sobel_y)
        
        edge = torch.tanh(torch.sqrt(edge_x.pow(2) + edge_y.pow(2)).view(b, c, h, w))
        return torch.cat([x, edge], dim=1)
        
    def forward(self, x):
        # Global feature path
        global_feat = self.global_conv(x)
        global_feat = self.layer_norm(global_feat.mean(dim=[2,3], keepdim=True)) * global_feat
        
        # Edge feature path
        edge_feat = self.get_edge_features(x)
        edge_feat = self.edge_conv(edge_feat)
        
        # Attention-based fusion
        attention = torch.sigmoid(self.fusion_conv(torch.cat([global_feat, edge_feat], dim=1)))
        feat = global_feat + edge_feat
        
        return attention * feat