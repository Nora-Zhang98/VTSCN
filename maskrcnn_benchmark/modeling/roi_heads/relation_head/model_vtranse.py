# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .utils_motifs import obj_edge_vectors, encode_box_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import TransformerEncoder
from .utils_relation import layer_init
from maskrcnn_benchmark.upt.interaction_head import InteractionHead
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import ScaledDotProductAttention
class VTransEFeature(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(VTransEFeature, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM

        # self.pred_layer = make_fc(self.obj_dim + self.embed_dim + 128, self.num_obj_classes)
        self.pred_layer = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.obj_ctx_compress = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim) 
        self.fc_layer = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)
        
        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim + self.embed_dim + 128))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.obj_dim + 128))

        self.tri_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRI_TRANSFORMER.LAYER
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRI_TRANSFORMER.NUM_HEAD
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRI_TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRI_TRANSFORMER.VAL_DIM
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRI_TRANSFORMER.INNER_DIM
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRI_TRANSFORMER.DROPOUT_RATE
        self.tri_encoder = TransformerEncoder(self.tri_layer, self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

        self.lin_tri = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_dim * 2 + 128, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim)  
        )
        layer_init(self.lin_tri[1], xavier=True)
        layer_init(self.lin_tri[4], xavier=True)
        # self.pred_embed_proj = nn.Linear(self.embed_dim, 128) 
        self.pred_embed_proj = nn.Linear(self.embed_dim, self.hidden_dim) 
        # ---------------UPT：InteractionHead for union feature-------------
        self.interaction_head = InteractionHead(self.cfg.UPT.HIDDEN_DIM, self.cfg.UPT.REPR_DIM, self.cfg.UPT.NUM_CHANNELS) 
        self.up_dim = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 8)

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, proposals, rel_pair_idxs, global_features, union_features, logger=None, all_average=False, ctx_average=False):
        num_objs = [len(b) for b in proposals]
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))

        batch_size = x.shape[0]
        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:
            obj_pre_rep = cat((x, obj_embed, pos_embed), -1)

        # -----------------UPT for union feature-----------------
        obj_ctx = self.obj_ctx_compress(obj_pre_rep)
        new_union_features = self.interaction_head(x, obj_ctx, proposals, global_features, rel_pair_idxs, self.mode)
        union_features = union_features + self.up_dim(new_union_features)  

        # object level contextual feature
        # obj_dists = self.pred_layer(obj_pre_rep)
        obj_dists = self.pred_layer(obj_ctx)
        obj_preds = obj_dists.max(-1)[1]

        num_objs = [len(p) for p in proposals]
        tri_reps = []
        obj_ctx_all = obj_ctx.split(num_objs)
        obj_label_all = obj_preds.split(num_objs)

        for obj_c, obj_l, rel_pair_id, num_obj in zip(obj_ctx_all, obj_label_all, rel_pair_idxs, num_objs):
            num_rel = len(rel_pair_id)
            if num_rel == 0: 
                continue
            sub_label = obj_l[rel_pair_id[:, 0]]
            obj_label = obj_l[rel_pair_id[:, 1]]
            sub_embed = self.obj_embed1(sub_label.long())
            obj_embed = self.obj_embed1(obj_label.long())
            tri_rep = self.pred_embed_proj(sub_embed - obj_embed)
            tri_rep = self.tri_encoder(tri_rep, [num_rel])
            tri_reps.append(tri_rep)
        tri_rep = cat(tri_reps)

        # edge level contextual feature

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_embed2 = F.softmax(obj_dists, dim=1) @ self.obj_embed2.weight
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_embed2), dim=-1)
        else:
            obj_embed2 = self.obj_embed2(obj_preds.long())
            obj_rel_rep = cat((x, pos_embed, obj_embed2), -1)
            
        edge_ctx = F.relu(self.fc_layer(obj_rel_rep))

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, cat((x, pos_embed), -1))

        return obj_dists, obj_preds, edge_ctx, None, tri_rep, union_features
