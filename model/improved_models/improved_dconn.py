import torch
import torch.nn as nn
from ..DconnNet import DconnNet, SpaceBlock
from .mrde import MRDE
from .glfi import GLFI
import torch.nn.functional as F

class ImprovedDconnNet(DconnNet):
    def __init__(self, num_class=1, decoder_attention=False, 
                 use_mrde=True, use_glfi=True, reduction_ratio=4):
        super().__init__(num_class, decoder_attention)
        
        self.use_mrde = use_mrde
        self.use_glfi = use_glfi
        
        # MRDE modules
        if self.use_mrde:
            self.mrde_blocks = nn.ModuleList([
                MRDE(64, reduction_ratio),    # After c1: 64 channels
                MRDE(64, reduction_ratio),    # After c2: 64 channels
                MRDE(128, reduction_ratio),   # After c3: 128 channels
                MRDE(256, reduction_ratio),   # After c4: 256 channels
                MRDE(512, reduction_ratio)    # For c5: 512 channels
            ])
            
        # GLFI modules and gates
        if self.use_glfi:
            self.glfi_blocks = nn.ModuleList([
                GLFI(256),  # For d4 (256 channels)
                GLFI(128),  # For d3 (128 channels)
                GLFI(64),   # For d2 (64 channels)
                GLFI(64)    # For d1 (64 channels)
            ])
            # Add gating modules for GLFI
            self.glfi_gates = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(256, 256, 1),
                    nn.BatchNorm2d(256)
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(128, 128, 1),
                    nn.BatchNorm2d(128)
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(64, 64, 1),
                    nn.BatchNorm2d(64)
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(64, 64, 1),
                    nn.BatchNorm2d(64)
                )
            ])
            
        # Channel adapters for skip connections
        self.skip_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 1),  # d4 -> sb1 (256->256)
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 1),  # d3 -> sb2 (128->256)
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.Conv2d(64, 256, 1),   # d2 -> sb3 (64->256)
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 1),   # d1 -> sb4 (64->128)
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False)
            )
        ])

        self.sb1 = SpaceBlock(512, 512, 512) # 输入512,中间512,输出512
        self.sb2 = SpaceBlock(512, 256, 256) # 输入512,中间256,输出256
        self.sb3 = SpaceBlock(256, 256, 128) # 输入256,中间256,输出128
        self.sb4 = SpaceBlock(128, 256, 64)  # 输入128(scene),中间256(content),输出64
    
    def residual_scaling(self, feat, residual, scale=0.05):
        """Dynamic scaling of residual features"""
        feat_norm = torch.norm(feat.view(feat.size(0), -1), p=2, dim=1, keepdim=True).view(feat.size(0), 1, 1, 1)
        residual_norm = torch.norm(residual.view(residual.size(0), -1), p=2, dim=1, keepdim=True).view(residual.size(0), 1, 1, 1)
        
        norm_ratio = feat_norm / (residual_norm + 1e-8)
        return feat + scale * norm_ratio * residual
    
    def forward(self, x):
        # Encoder path
        c1 = self.backbone.conv1(x)
        c1 = self.backbone.bn1(c1)
        c1 = self.backbone.relu(c1)
        
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)

        # Apply MRDE
        if self.use_mrde:
            features = [c1, c2, c3, c4]
            for i in range(len(features)):
                features[i] = self.mrde_blocks[i](features[i])
            c1, c2, c3, c4 = features

        # Original DconnNet forward path
        directional_c5 = self.channel_mapping(c5)
        mapped_c5 = F.interpolate(directional_c5, scale_factor=32, mode='bilinear', align_corners=True)
        mapped_c5 = self.direc_reencode(mapped_c5)
        
        d_prior = self.gap(mapped_c5)
        
        if self.use_mrde:
            c5 = self.mrde_blocks[4](c5)
        else:
            c5 = self.sde_module(c5, d_prior)
        
        c6 = self.gap(c5)
        
        # Space blocks with skip connections
        r5 = self.sb1(c6, c5)
        
        d4 = self.relu(self.fb5(r5) + c4)
        if self.use_glfi:
            glfi_feat = self.glfi_blocks[0](d4)
            gate = torch.sigmoid(self.glfi_gates[0](d4))
            d4 = self.residual_scaling(d4, gate * glfi_feat)
        r4 = self.sb2(self.gap(r5), self.skip_adapters[0](d4))
        
        d3 = self.relu(self.fb4(r4) + c3)
        if self.use_glfi:
            glfi_feat = self.glfi_blocks[1](d3)
            gate = torch.sigmoid(self.glfi_gates[1](d3))
            d3 = self.residual_scaling(d3, gate * glfi_feat)
        r3 = self.sb3(self.gap(r4), self.skip_adapters[1](d3))
        
        d2 = self.relu(self.fb3(r3) + c2)
        if self.use_glfi:
            glfi_feat = self.glfi_blocks[2](d2)
            gate = torch.sigmoid(self.glfi_gates[2](d2))
            d2 = self.residual_scaling(d2, gate * glfi_feat)
        r2 = self.sb4(self.gap(r3), self.skip_adapters[2](d2))
        
        d1 = self.fb2(r2) + c1
        if self.use_glfi:
            glfi_feat = self.glfi_blocks[3](d1)
            gate = torch.sigmoid(self.glfi_gates[3](d1))
            d1 = self.residual_scaling(d1, gate * glfi_feat)
            
        # Final decoding
        feat_list = [d1, d2, d3, d4]
        
        attns = None
        if self.decoder_attention:
            attns = [c1, 
                    nn.UpsamplingBilinear2d(scale_factor=2)(c2),
                    nn.UpsamplingBilinear2d(scale_factor=4)(c3),
                    nn.UpsamplingBilinear2d(scale_factor=8)(c4)]
            attns = self.attention_producer(torch.cat(attns, dim=1))
            n, c, h, w = attns.shape
            attns = torch.split(attns.reshape(n, c // 4, 4, h, w), split_size_or_sections=1, dim=1)
            attns = torch.concat([F.softmax(attn, dim=2) for attn in attns], dim=1)
            
        final_feat = self.final_decoder(feat_list, attns=attns)
        cls_pred = self.cls_pred_conv_2(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)
        
        return cls_pred, mapped_c5