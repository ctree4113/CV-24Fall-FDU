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
        base_loss = self.base_criterion(pred, target)
        
        # Frequency loss
        freq_loss = self.frequency_loss(pred, target)
        
        # Topology consistency loss
        topo_loss = self.topology_loss(pred, target)
        
        # Combined loss
        total_loss = base_loss + self.freq_weight * freq_loss + self.topo_weight * topo_loss
        
        return total_loss
    
    def frequency_loss(self, pred, target):
        """Calculate frequency domain loss"""
        # Convert to frequency domain
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Calculate magnitude spectrum
        pred_magnitude = torch.abs(pred_fft)
        target_magnitude = torch.abs(target_fft)
        
        # Calculate L1 loss in frequency domain
        freq_loss = F.l1_loss(pred_magnitude, target_magnitude)
        
        return freq_loss
    
    def topology_loss(self, pred, target):
        """Calculate topology consistency loss"""
        # Use Sobel operator to calculate gradients
        pred_grad_x = self.sobel_gradient_x(pred)
        pred_grad_y = self.sobel_gradient_y(pred)
        target_grad_x = self.sobel_gradient_x(target)
        target_grad_y = self.sobel_gradient_y(target)
        
        # Calculate gradient direction
        pred_direction = torch.atan2(pred_grad_y, pred_grad_x)
        target_direction = torch.atan2(target_grad_y, target_grad_x)
        
        # Calculate direction difference
        direction_diff = 1 - torch.cos(pred_direction - target_direction)
        
        # Calculate gradient magnitude
        pred_magnitude = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        target_magnitude = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        # Use gradient magnitude as weight
        weights = torch.min(pred_magnitude, target_magnitude)
        
        # Calculate weighted topology loss
        topo_loss = (direction_diff * weights).mean()
        
        return topo_loss
    
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