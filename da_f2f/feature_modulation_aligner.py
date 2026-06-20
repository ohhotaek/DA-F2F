import torch
import torch.nn.functional as F
import torch.nn as nn

from detectron2.config import configurable
from detectron2.modeling import GeneralizedRCNN

from da_f2f.utils import SaveIO, grad_reverse


class StyleModulator(nn.Module):
    """Style modulator (FiLM-based: gamma * feat + beta from target stats)"""
    def __init__(self, in_channels=256):
        super().__init__()
        self.in_channels = in_channels
        # Takes statistics (mu, sigma) as input -> 2 * C
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels * 2)
        )
        # Initialize last layer to 0 -> ensures initial gamma=1.0, beta=0.0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, style_vec):
        # style_vec: [1, 2*C]
        out = self.mlp(style_vec)
        gamma, beta = out.chunk(2, dim=-1)
        gamma = torch.clamp(gamma, min=-0.5, max=0.5) 
        beta = torch.clamp(beta, min=-1.0, max=1.0)
        # Ensure identity mapping
        C = gamma.shape[-1]
        gamma = (gamma + 1.0).view(-1, C, 1, 1)
        beta = beta.view(-1, C, 1, 1)
        return gamma, beta


class AlignMixin(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        rpn_da_enabled: bool = False,
        rpn_da_weight: float = 0.0,
        rpn_da_input_dim: int = 256,
        rpn_da_hidden_dims: list = [256,],
        style_enabled: bool = False,      
        style_alpha: float = 0.3,          
        style_levels: list = ["p2", "p3"],
        **kwargs
    ):
        super(AlignMixin, self).__init__(**kwargs)
        
        # Configuration
        self.rpn_da_weight = rpn_da_weight
        
        # Style modulator for style transfer
        self.style_enabled = style_enabled
        self.style_alpha = style_alpha
        self.style_levels = style_levels if style_levels is not None else ["p2", "p3"]
        self.style_gen = StyleModulator(in_channels=256) if style_enabled else None
        self._cached_target_style = None
        
        # Domain discriminators
        self.rpn_align = SpatialDomainHead(rpn_da_input_dim) if rpn_da_enabled else None
        
        # RPN DA configuration
        self.fore_weight = 0.8
        self.rpn_da_features = []
        self.current_domain_label = None
        self._rpn_obj_logits = None
        self._src_gt_boxes = None
        self._img_sizes = None
        
        # Register hooks
        self.backbone_io, self.rpn_io, self.roih_io, self.boxhead_io = SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.backbone.register_forward_hook(self.backbone_io)
        self.backbone.register_forward_hook(self._backbone_hook) 
        self.proposal_generator.register_forward_hook(self.rpn_io)
        self.roi_heads.register_forward_hook(self.roih_io)
        self.roi_heads.register_forward_pre_hook(self._roi_heads_pre_hook)
        
        # RPN hooks
        def _rpn_conv_hook(module, input, output):
            if self.training and self.rpn_align and self.current_domain_label is not None:
                self.rpn_da_features.append(output)
        
        def _rpn_head_hook(module, inputs, outputs):
            if not (self.training and self.current_domain_label is not None):
                return
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
                self._rpn_obj_logits = outputs[0]
            elif isinstance(outputs, dict) and "pred_objectness_logits" in outputs:
                self._rpn_obj_logits = outputs["pred_objectness_logits"]
            else:
                self._rpn_obj_logits = None
        
        if rpn_da_enabled:
            self.proposal_generator.rpn_head.conv.register_forward_hook(_rpn_conv_hook)
            self.proposal_generator.rpn_head.register_forward_hook(_rpn_head_hook)

    @classmethod
    def from_config(cls, cfg):
        ret = super(AlignMixin, cls).from_config(cfg)
        
        ret.update({
            "rpn_da_enabled": cfg.DA.ALIGN.RPN_DA_ENABLED,
            "rpn_da_weight": cfg.DA.ALIGN.RPN_DA_WEIGHT,
            "rpn_da_input_dim": cfg.DA.ALIGN.RPN_DA_INPUT_DIM,
            "rpn_da_hidden_dims": cfg.DA.ALIGN.RPN_DA_HIDDEN_DIMS,
            "style_enabled": cfg.DA.ALIGN.STYLE_ENABLED,    
            "style_alpha": cfg.DA.ALIGN.STYLE_ALPHA,          
            "style_levels": cfg.DA.ALIGN.STYLE_LEVELS,
        })
        
        return ret
    
    def _extract_style_stats(self, features):
        """Extract style statistics from p2, p3 features"""
        stats_dict = {}
        for lvl in self.style_levels:
            if lvl not in features:  
                continue
            feat = features[lvl]
            mu = feat.mean(dim=(2, 3))    # [B, C]
            var = feat.var(dim=(2, 3), unbiased=False)
            sigma = (var + 1e-5).sqrt()
            # Batch average and detach
            stats_dict[lvl] = torch.cat([mu, sigma], dim=1).mean(dim=0, keepdim=True).detach()
        return stats_dict
        
    def _backbone_hook(self, module, inputs, outputs):  
        """Hook to capture backbone outputs - no-op, just for compatibility"""
        pass
    
    def _roi_heads_pre_hook(self, module, inputs):
        """Apply style modulation before ROI heads (after RPN)"""
        from detectron2.utils.comm import is_main_process
        
        if not (self.training and 
                self.style_enabled and 
                getattr(self, "current_domain_label", None) == 1.0 and 
                self._cached_target_style is not None):
            return inputs
        
        # inputs: (images, features, proposals, gt_instances)
        images, features, proposals, gt_instances = inputs
        
        # Create modified features for ROI (RPN already used original features)
        new_features = {k: v for k, v in features.items()}
        
        for lvl in ["p2", "p3"]:
            if lvl in self._cached_target_style and lvl in features:
                gamma, beta = self.style_gen(self._cached_target_style[lvl])
                self._last_gamma = gamma
                self._last_beta = beta
                transformed = gamma * features[lvl] + beta
                new_features[lvl] = features[lvl] + self.style_alpha * (transformed - features[lvl])
                if torch.isnan(new_features[lvl]).any():
                    new_features[lvl] = features[lvl]
    
        
        self._last_roi_input_features = new_features
        return (images, new_features, proposals, gt_instances)
        
    def _compute_rpn_da_loss(self, domain_label, device):
        """Compute RPN domain adaptation loss with foreground/background weighting"""
        if len(self.rpn_da_features) == 0:
            return None
        
        sorted_feats = sorted(self.rpn_da_features, key=lambda f: f.shape[2] * f.shape[3], reverse=True)
        level_weights = [1.0, 1.0, 0.5, 0.5]
        if len(sorted_feats) > len(level_weights):
            level_weights = level_weights + [0.5] * (len(sorted_feats) - len(level_weights))
        
        total_loss = torch.tensor(0.0, device=device)
        total_weight = 0.0
        
        fg_w = self.fore_weight
        bg_w = 0.05
        
        for level_idx, feat in enumerate(sorted_feats):
            if torch.isnan(feat).any() or torch.isinf(feat).any():
                continue
            
            lw = level_weights[level_idx]
            rpn_reverse = grad_reverse(feat)
            if torch.isnan(rpn_reverse).any() or torch.isinf(rpn_reverse).any():
                continue
            
            rpn_domain_map = self.rpn_align(rpn_reverse)
            rpn_domain_map = torch.clamp(rpn_domain_map, min=-10, max=10)
            domain_label_map = torch.full_like(rpn_domain_map, domain_label)
            
            loss_map = F.binary_cross_entropy_with_logits(
                rpn_domain_map, domain_label_map, reduction="none"
            )
            
            B = loss_map.shape[0]
            
            if self._src_gt_boxes and len(self._src_gt_boxes) > 0 and domain_label == 1:
                # Source: separate fg/bg per batch
                feat_h, feat_w = rpn_domain_map.shape[2], rpn_domain_map.shape[3]
                level_loss = torch.tensor(0.0, device=device)
                
                for b in range(B):
                    img_h, img_w = self._img_sizes[b]
                    scale_y = feat_h / img_h
                    scale_x = feat_w / img_w
                    
                    loss_b = loss_map[b]
                    fg_mask_b = torch.zeros_like(loss_b)
                    
                    if b < len(self._src_gt_boxes):
                        for box in self._src_gt_boxes[b]:
                            x1 = max(0, min(int(box[0].item() * scale_x), feat_w))
                            y1 = max(0, min(int(box[1].item() * scale_y), feat_h))
                            x2 = max(0, min(int(box[2].item() * scale_x), feat_w))
                            y2 = max(0, min(int(box[3].item() * scale_y), feat_h))
                            if x2 > x1 and y2 > y1:
                                fg_mask_b[:, y1:y2, x1:x2] = 1.0
                    
                    bg_mask_b = 1.0 - fg_mask_b
                    fg_count = fg_mask_b.sum()
                    bg_count = bg_mask_b.sum()
                    
                    loss_fg = (loss_b * fg_mask_b).sum() / (fg_count + 1e-6) if fg_count > 0 else torch.tensor(0.0, device=device)
                    loss_bg = (loss_b * bg_mask_b).sum() / (bg_count + 1e-6)
                    
                    level_loss = level_loss + fg_w * loss_fg + bg_w * loss_bg
                
                level_loss = level_loss / B
            
            else:
                # Target: use objectness logits for weighting
                obj_logit = None
                if self._rpn_obj_logits is not None:
                    for ol in self._rpn_obj_logits:
                        if ol.shape[2] == feat.shape[2] and ol.shape[3] == feat.shape[3]:
                            obj_logit = ol
                            break
                
                if obj_logit is not None:
                    p_obj = torch.sigmoid(obj_logit.max(dim=1, keepdim=True)[0]).detach()
                    weight_map = bg_w + (fg_w - bg_w) * p_obj
                else:
                    weight_map = torch.ones_like(rpn_domain_map)
                
                level_loss = (loss_map * weight_map).sum() / (weight_map.sum() + 1e-6)
            
            if not torch.isnan(level_loss):
                total_loss = total_loss + lw * level_loss
                total_weight += lw
        
        return total_loss / total_weight if total_weight > 0 else None
    
    def forward(self, *args, do_align=False, alpha=1.0, pred_diff=None, pseudo_boxes=None, **kwargs):
        self.current_domain_label = alpha if (self.training and do_align) else None
        batched_inputs = args[0]
        self.rpn_da_features = []  # Reset
        
        # Single forward pass: RPN (original) -> Pre-hook (Style) -> ROI (transformed)
        output = super().forward(*args, **kwargs)
        
        if self.training:
            self._img_sizes = [(s['height'], s['width']) for s in batched_inputs]
            
            # Cache target style when processing target domain (alpha=0.0)
            if do_align and alpha == 0.0:
                features = self.backbone_io.output
                if "p2" in features and "p3" in features:
                    self._cached_target_style = self._extract_style_stats(features)
            
            # Save source GT boxes when processing source domain (alpha=1.0)
            elif do_align and alpha == 1.0:
                self._src_gt_boxes = [
                    s['instances'].gt_boxes.tensor.cuda() if 'instances' in s 
                    else torch.zeros((0, 4)).cuda() 
                    for s in batched_inputs
                ]
            
            if do_align:
                domain_label = alpha
                device = list(self.backbone_io.output.values())[0].device
                
                # Dummy loss to keep style modulator in computation graph
                if self.style_gen is not None:
                    fake_output = sum([p.sum() for p in self.style_gen.parameters()]) * 0
                    output["_da_style"] = fake_output
                
                # RPN-level DA
                if self.rpn_align:
                    rpn_loss = self._compute_rpn_da_loss(domain_label, device)
                    if rpn_loss is not None:
                        output["loss_da_rpn"] = self.rpn_da_weight * rpn_loss
            
            elif self.rpn_align or self.style_gen:
                # Keep all modules in computation graph
                fake_output = 0
                for aligner in [self.rpn_align, self.style_gen]:
                    if aligner is not None:
                        fake_output += sum([p.sum() for p in aligner.parameters()]) * 0
                output["_da"] = fake_output
                
        return output


class SpatialDomainHead(torch.nn.Module):
    """Spatial domain classifier (BDC-FR style)"""
    def __init__(self, input_dim=256):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, 256, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1, 1),  # Output: [B, 1, H, W]
        )
    
    def forward(self, x):
        return self.head(x)