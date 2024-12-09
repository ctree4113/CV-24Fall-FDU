import torch
import torch.nn as nn
import torch.nn.functional as F

class ConnectLoss(nn.Module):
    """Base connectivity loss from original DconnNet"""
    def __init__(self, num_class=1):
        super().__init__()
        self.num_class = num_class
        self.bce_loss = nn.BCELoss(reduction='none')
        self.dice_loss = DiceLoss()
        
    def connectivity_matrix(self, multimask):
        """Convert segmentation masks to connectivity masks"""
        [batch, _, rows, cols] = multimask.shape
        conn = torch.zeros([batch, self.num_class*8, rows, cols]).cuda()
        
        for i in range(self.num_class):
            mask = multimask[:, i, :, :]
            
            # Generate 8-directional connectivity maps
            up = torch.zeros_like(mask)
            down = torch.zeros_like(mask)
            left = torch.zeros_like(mask)
            right = torch.zeros_like(mask)
            up_left = torch.zeros_like(mask)
            up_right = torch.zeros_like(mask)
            down_left = torch.zeros_like(mask)
            down_right = torch.zeros_like(mask)

            # Shift masks in 8 directions
            up[:, :rows-1, :] = mask[:, 1:rows, :]
            down[:, 1:rows, :] = mask[:, 0:rows-1, :]
            left[:, :, :cols-1] = mask[:, :, 1:cols]
            right[:, :, 1:cols] = mask[:, :, :cols-1]
            up_left[:, 0:rows-1, 0:cols-1] = mask[:, 1:rows, 1:cols]
            up_right[:, 0:rows-1, 1:cols] = mask[:, 1:rows, 0:cols-1]
            down_left[:, 1:rows, 0:cols-1] = mask[:, 0:rows-1, 1:cols]
            down_right[:, 1:rows, 1:cols] = mask[:, 0:rows-1, 0:cols-1]

            # Compute connectivity maps
            conn[:, (i*8)+0, :, :] = mask * down_right
            conn[:, (i*8)+1, :, :] = mask * down
            conn[:, (i*8)+2, :, :] = mask * down_left
            conn[:, (i*8)+3, :, :] = mask * right
            conn[:, (i*8)+4, :, :] = mask * left
            conn[:, (i*8)+5, :, :] = mask * up_right
            conn[:, (i*8)+6, :, :] = mask * up
            conn[:, (i*8)+7, :, :] = mask * up_left

        return conn.float()

    def bilateral_voting(self, c_map, hori_translation, verti_translation):
        """Bilateral voting to convert connectivity maps to segmentation"""
        batch, class_num, channel, row, column = c_map.size()
        vote_out = torch.zeros([batch, class_num, channel, row, column]).cuda()
        
        # Apply voting in each direction
        # Right direction
        right = torch.bmm(c_map[:,:,4].contiguous().view(-1,row,column), 
                         hori_translation.view(-1,column,column))
        right = right.view(batch,class_num,row,column)

        # Left direction
        left = torch.bmm(c_map[:,:,3].contiguous().view(-1,row,column),
                        hori_translation.transpose(3,2).view(-1,column,column))
        left = left.view(batch,class_num,row,column)

        # Bottom direction
        bottom = torch.bmm(verti_translation.transpose(3,2).view(-1,row,row),
                          c_map[:,:,6].contiguous().view(-1,row,column))
        bottom = bottom.view(batch,class_num,row,column)

        # Up direction
        up = torch.bmm(verti_translation.view(-1,row,row),
                      c_map[:,:,1].contiguous().view(-1,row,column))
        up = up.view(batch,class_num,row,column)

        # Left-bottom direction
        left_bottom = torch.bmm(verti_translation.transpose(3,2).view(-1,row,row),
                               c_map[:,:,5].contiguous().view(-1,row,column))
        left_bottom = torch.bmm(left_bottom.view(-1,row,column),
                               hori_translation.transpose(3,2).view(-1,column,column))
        left_bottom = left_bottom.view(batch,class_num,row,column)

        # Right-above direction
        right_above = torch.bmm(verti_translation.view(-1,row,row),
                               c_map[:,:,2].contiguous().view(-1,row,column))
        right_above = torch.bmm(right_above.view(-1,row,column),
                               hori_translation.view(-1,column,column))
        right_above = right_above.view(batch,class_num,row,column)

        # Left-above direction
        left_above = torch.bmm(verti_translation.view(-1,row,row),
                              c_map[:,:,0].contiguous().view(-1,row,column))
        left_above = torch.bmm(left_above.view(-1,row,column),
                              hori_translation.transpose(3,2).view(-1,column,column))
        left_above = left_above.view(batch,class_num,row,column)

        # Right-bottom direction
        right_bottom = torch.bmm(verti_translation.transpose(3,2).view(-1,row,row),
                                c_map[:,:,7].contiguous().view(-1,row,column))
        right_bottom = torch.bmm(right_bottom.view(-1,row,column),
                                hori_translation.view(-1,column,column))
        right_bottom = right_bottom.view(batch,class_num,row,column)

        # Combine votes
        vote_out[:,:,0] = (c_map[:,:,0]) * (right_bottom)
        vote_out[:,:,1] = (c_map[:,:,1]) * (bottom)
        vote_out[:,:,2] = (c_map[:,:,2]) * (left_bottom)
        vote_out[:,:,3] = (c_map[:,:,3]) * (right)
        vote_out[:,:,4] = (c_map[:,:,4]) * (left)
        vote_out[:,:,5] = (c_map[:,:,5]) * (right_above)
        vote_out[:,:,6] = (c_map[:,:,6]) * (up)
        vote_out[:,:,7] = (c_map[:,:,7]) * (left_above)

        pred_mask, _ = torch.max(vote_out, dim=2)
        return pred_mask, vote_out

    def edge_loss(self, vote_out, edge):
        """Compute edge consistency loss"""
        pred_mask_min, _ = torch.min(vote_out.cuda(), dim=2)
        pred_mask_min = pred_mask_min * edge
        min_loss = self.bce_loss(pred_mask_min, 
                                torch.full_like(pred_mask_min, 0))
        return (min_loss.sum() / pred_mask_min.sum())

    def forward(self, pred, target):
        """
        Args:
            pred: Model prediction (B, C*8, H, W)
            target: Ground truth mask (B, C, H, W)
        """
        # Get connectivity maps
        conn_target = self.connectivity_matrix(target)
        
        # Compute connectivity loss
        conn_loss = self.bce_loss(F.sigmoid(pred), conn_target).mean()
        
        # Get edge maps
        class_conn = conn_target.view([pred.shape[0], self.num_class, 8,
                                     pred.shape[2], pred.shape[3]])
        sum_conn = torch.sum(class_conn, dim=2)
        edge = torch.where((sum_conn < 8) & (sum_conn > 0),
                          torch.full_like(sum_conn, 1),
                          torch.full_like(sum_conn, 0))
        
        # Compute edge loss
        edge_loss = self.edge_loss(F.sigmoid(
            pred.view([pred.shape[0], self.num_class, 8,
                      pred.shape[2], pred.shape[3]])), edge)
        
        # Compute segmentation loss
        class_pred = pred.view([pred.shape[0], self.num_class, 8,
                              pred.shape[2], pred.shape[3]])
        final_pred, vote_out = self.bilateral_voting(F.sigmoid(class_pred),
                                                   self.hori_translation,
                                                   self.verti_translation)
        seg_loss = self.dice_loss(final_pred, target)
        
        # Combine losses
        total_loss = conn_loss + edge_loss + seg_loss
        
        return total_loss

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean() 