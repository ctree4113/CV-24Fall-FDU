import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedLoss(nn.Module):
    """Improved loss function combining base loss with frequency and topology consistency"""
    def __init__(self, base_criterion, freq_weight=0.05, topo_weight=0.05):
        super().__init__()
        self.base_criterion = base_criterion
        self.freq_weight = freq_weight
        self.topo_weight = topo_weight
        
    def forward(self, pred, target):
        # 基础损失
        base_loss = self.base_criterion(pred, target)
        
        # 频率域损失
        freq_loss = self.frequency_loss(pred, target)
        
        # 拓扑一致性损失
        topo_loss = self.topology_loss(pred, target)
        
        # 组合损失
        total_loss = base_loss + self.freq_weight * freq_loss + self.topo_weight * topo_loss
        
        return total_loss
    
    def frequency_loss(self, pred, target):
        """计算频率域损失"""
        # 转换为频率域
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # 计算幅度谱
        pred_magnitude = torch.abs(pred_fft)
        target_magnitude = torch.abs(target_fft)
        
        # 计算频率域的L1损失
        freq_loss = F.l1_loss(pred_magnitude, target_magnitude)
        
        return freq_loss
    
    def topology_loss(self, pred, target):
        """计算拓扑一致性损失"""
        # 使用Sobel算子计算梯度
        pred_grad_x = self.sobel_gradient_x(pred)
        pred_grad_y = self.sobel_gradient_y(pred)
        target_grad_x = self.sobel_gradient_x(target)
        target_grad_y = self.sobel_gradient_y(target)
        
        # 计算梯度方向
        pred_direction = torch.atan2(pred_grad_y, pred_grad_x)
        target_direction = torch.atan2(target_grad_y, target_grad_x)
        
        # 计算方向差异
        direction_diff = 1 - torch.cos(pred_direction - target_direction)
        
        # 计算梯度幅值
        pred_magnitude = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        target_magnitude = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        # 使用梯度幅值作为权重
        weights = torch.min(pred_magnitude, target_magnitude)
        
        # 计算加权拓扑损失
        topo_loss = (direction_diff * weights).mean()
        
        return topo_loss
    
    def sobel_gradient_x(self, x):
        """Sobel X方向梯度"""
        sobel_x = torch.tensor([[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        return F.conv2d(x, sobel_x, padding=1)
    
    def sobel_gradient_y(self, x):
        """Sobel Y方向梯度"""
        sobel_y = torch.tensor([[-1, -2, -1], 
                              [0, 0, 0], 
                              [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        return F.conv2d(x, sobel_y, padding=1)