# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_intersection
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor



@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, cfg, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separete spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim//2), nn.ReLU(inplace=True),
                                              make_fc(out_dim//2, out_dim), nn.ReLU(inplace=True),
                                            ])

        # union rectangle size
        self.rect_size = resolution * 4 -1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            ])
        

    def forward(self, x, proposals, rel_pair_idxs=None):
        device = x[0].device
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposal = boxlist_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
            # resize bbox to the scale rect_size
            head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
            tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
            head_rect = ((dummy_x_range >= head_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= head_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= head_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= head_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= tail_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= tail_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= tail_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()

            rect_input = torch.stack((head_rect, tail_rect), dim=1) # (num_rel, 4, rect_size, rect_size)
            rect_inputs.append(rect_input)

        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        rect_inputs = torch.cat(rect_inputs, dim=0)
        rect_features = self.rect_conv(rect_inputs)

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)
        # merge two parts
        if self.separate_spatial:
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_features = union_vis_features + rect_features
            union_features = self.shift(union_features, shift_div=8) 
            union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)
            
        return union_features

    def shift(self, x, shift_div=5):
        """Perform temporal shift operation on the feature.

        Args:
            x (torch.Tensor): The input feature to be shifted.
            shift_div (int): Number of divisions for shift. Default: 8.

        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, C, H, W] 
        n, c, h, w = x.size()

        # [N // num_segments, num_segments, C, H,W]

        # x = x.view(-1, c, h, w) 

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        H_f_split = x[:, :fold, :, :]
        H_b_split = x[:, fold:2 * fold, :, :]
        W_f_split = x[:, 2 * fold:3 * fold, :, :]
        W_b_split = x[:, 3 * fold:4 * fold, :, :]
        no_shift_split = x[:, 4 * fold:, :, :]

        # shift Height forward on num_segments channel in `H_f_split`
        zeros = H_f_split - H_f_split
        blank = zeros[:, :, :1, :]
        H_f_split = H_f_split[:, :, 1:, :]  
        H_f_split = torch.cat((H_f_split, blank), 2)

        # shift Height backward on num_segments channel in `H_f_split`
        zeros = H_b_split - H_b_split
        blank = zeros[:, :, :1, :]
        H_b_split = H_b_split[:, :, :-1, :]  
        H_b_split = torch.cat((blank, H_b_split), 2)

        # shift Width forward on num_segments channel in `W_f_split`
        zeros = W_f_split - W_f_split
        blank = zeros[:, :, :, :1]
        W_f_split = W_f_split[:, :, :, 1:]  
        W_f_split = torch.cat((W_f_split, blank), 3)

        # shift Width backward on num_segments channel in `W_b_split`
        zeros = W_b_split - W_b_split
        blank = zeros[:, :, :, :1]
        W_b_split = W_b_split[:, :, :, :-1]  
        W_b_split = torch.cat((blank, W_b_split), 3)

        # no_shift_split: no shift

        # concatenate
        out = torch.cat((H_f_split, H_b_split, W_f_split, W_b_split, no_shift_split), 1)

        # [N, C, H, W]
        # restore the original dimension
        return out.view(n, c, h, w)

def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
