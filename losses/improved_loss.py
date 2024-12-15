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
        # 首先计算基础损失
        base_loss = self.base_criterion(pred, target)
        
        # 对于频域损失和拓扑损失，我们需要先处理预测输出
        if len(pred.shape) == 4 and pred.shape[1] == 8:  # (B, 8, H, W)
            # 对于方向连接预测，我们取平均值作为最终预测
            pred_processed = torch.sigmoid(pred).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            pred_processed = torch.sigmoid(pred)
        
        # 确保 target 的维度正确
        if len(target.shape) == 3:
            target = target.unsqueeze(1)  # (B, 1, H, W)
            
        # 计算频域损失
        freq_loss = self.frequency_loss(pred_processed, target)
        
        # 计算拓扑损失
        topo_loss = self.topology_loss(pred_processed, target)
        
        # 组合损失
        total_loss = base_loss + self.freq_weight * freq_loss + self.topo_weight * topo_loss
        
        return total_loss
    
    def frequency_loss(self, pred, target):
        """Calculate frequency domain loss"""
        # 确保输入在合理范围内
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        target = torch.clamp(target, min=1e-7, max=1-1e-7)
        
        # 转换到频域
        pred_fft = torch.fft.fft2(pred.float())
        target_fft = torch.fft.fft2(target.float())
        
        # 计算幅度谱
        pred_magnitude = torch.abs(pred_fft) + 1e-7  # 添加小值避免取对数时出现问题
        target_magnitude = torch.abs(target_fft) + 1e-7
        
        # 计算对数幅度谱的差异
        freq_loss = F.l1_loss(
            torch.log(pred_magnitude), 
            torch.log(target_magnitude)
        )
        
        return freq_loss
    
    def topology_loss(self, pred, target):
        """Calculate topology consistency loss"""
        # 确保输入在合理范围内
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        target = torch.clamp(target, min=1e-7, max=1-1e-7)
        
        # 计算每个通道的梯度
        batch_size = pred.shape[0]
        total_loss = 0
        
        for b in range(batch_size):
            # 提取单个样本
            pred_sample = pred[b]  # (C, H, W)
            target_sample = target[b]  # (C, H, W)
            
            # 对每个通道计算梯度
            for c in range(pred_sample.shape[0]):
                # 计算 Sobel 梯度
                pred_grad_x = self.sobel_gradient_x(pred_sample[c:c+1].unsqueeze(0))
                pred_grad_y = self.sobel_gradient_y(pred_sample[c:c+1].unsqueeze(0))
                target_grad_x = self.sobel_gradient_x(target_sample[c:c+1].unsqueeze(0))
                target_grad_y = self.sobel_gradient_y(target_sample[c:c+1].unsqueeze(0))
                
                # 计算梯度方向
                pred_direction = torch.atan2(pred_grad_y + 1e-7, pred_grad_x + 1e-7)
                target_direction = torch.atan2(target_grad_y + 1e-7, target_grad_x + 1e-7)
                
                # 计算梯度幅值
                pred_magnitude = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-7)
                target_magnitude = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-7)
                
                # 计算方向一致性损失
                direction_diff = 1 - torch.cos(pred_direction - target_direction)
                
                # 使用梯度幅值作为权重
                weights = torch.min(pred_magnitude, target_magnitude)
                
                # 累加加权的拓扑损失
                total_loss += (direction_diff * weights).mean()
        
        return total_loss / (batch_size * pred.shape[1])
    
    def sobel_gradient_x(self, x):
        """Sobel gradient in X direction"""
        sobel_x = torch.tensor([[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        return F.conv2d(x, sobel_x, padding=1)
    
    def sobel_gradient_y(self, x):
        """Sobel gradient in Y direction"""
        sobel_y = torch.tensor([[-1, -2, -1], 
                              [0, 0, 0], 
                              [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        return F.conv2d(x, sobel_y, padding=1)