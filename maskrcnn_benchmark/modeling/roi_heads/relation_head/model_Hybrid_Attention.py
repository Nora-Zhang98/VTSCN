'''Rectified Identity Cell'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_co_attention import Self_Attention_Encoder, Cross_Attention_Encoder
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors,\
    to_onehot, nms_overlaps, encode_box_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import TransformerEncoder
from .utils_relation import layer_init
from maskrcnn_benchmark.upt.interaction_head import InteractionHead

class Self_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Self_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.SA_transformer_encoder = Self_Attention_Encoder(self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats=None, num_objs=None):
        assert num_objs is not None
        outp = self.SA_transformer_encoder(x, num_objs)

        return outp

class Cross_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Cross_Attention_Cell, self).__init__()
        self.cfg = config
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.CA_transformer_encoder = Cross_Attention_Encoder(self.num_head, self.k_dim,
                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)


    def forward(self, x, textual_feats, num_objs=None):
        assert num_objs is not None
        outp = self.CA_transformer_encoder(x, textual_feats, num_objs)

        return outp

class Single_Layer_Hybrid_Attention(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.SA_Cell_vis = Self_Attention_Cell(config)
        self.SA_Cell_txt = Self_Attention_Cell(config)
        self.CA_Cell_vis = Cross_Attention_Cell(config)
        self.CA_Cell_txt = Cross_Attention_Cell(config)

    def forward(self, visual_feats, text_feats, num_objs):
        tsa = self.SA_Cell_txt(text_feats, num_objs=num_objs)
        tca = self.CA_Cell_txt(text_feats, visual_feats, num_objs=num_objs)
        vsa = self.SA_Cell_vis(visual_feats, num_objs=num_objs)
        vca = self.CA_Cell_vis(visual_feats, text_feats, num_objs=num_objs)
        textual_output = tsa + tca
        visual_output = vsa + vca

        return visual_output, textual_output

class SHA_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config, n_layers):
        super().__init__()
        self.cfg = config
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD # 8头
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM # 2048
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM # 512
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM # 64
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM # 64
        self.cross_module = nn.ModuleList([
            Single_Layer_Hybrid_Attention(config)
            for _ in range(n_layers)])

    def forward(self, visual_feats, text_feats, num_objs):
        visual_output = visual_feats
        textual_output = text_feats

        for enc_layer in self.cross_module:
            visual_output, textual_output = enc_layer(visual_output, textual_output, num_objs)

        visual_output = visual_output + textual_output

        return visual_output, textual_output

class SHA_Context(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM # glove：200
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM # 512
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER  # 4
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER # 2

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj_visual = nn.Linear(self.in_channels + 128, self.hidden_dim)
        self.lin_obj_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.lin_edge_visual = nn.Linear(self.hidden_dim + self.in_channels, self.hidden_dim)
        self.lin_edge_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)

        self.context_obj = SHA_Encoder(config, self.obj_layer)
        self.context_edge = SHA_Encoder(config, self.edge_layer)

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
            nn.Linear(self.hidden_dim, self.hidden_dim)  # 最后输出维度是512
        )
        layer_init(self.lin_tri[1], xavier=True)
        layer_init(self.lin_tri[4], xavier=True)
        self.pred_embed_proj = nn.Linear(self.embed_dim, self.hidden_dim)  # 200维映射到512 新
        # ---------------UPT：InteractionHead提取union feature-------------
        self.interaction_head = InteractionHead(self.cfg.UPT.HIDDEN_DIM, self.cfg.UPT.REPR_DIM,
                                                self.cfg.UPT.NUM_CHANNELS)  # 256,512,2048
        self.up_dim = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 8)

    def forward(self, roi_features, proposals, global_features, rel_pair_idxs, union_features, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer

        num_objs = [len(p) for p in proposals]
        obj_pre_rep_vis = cat((roi_features, pos_embed), -1) # 116,1024
        obj_pre_rep_vis = self.lin_obj_visual(obj_pre_rep_vis) # 4224→512
        obj_pre_rep_txt = obj_embed
        obj_pre_rep_txt = self.lin_obj_textual(obj_pre_rep_txt) # 116,1024
        obj_feats_vis, _, = self.context_obj(obj_pre_rep_vis, obj_pre_rep_txt, num_objs) # SHA encoder，只保留第一个返回值，即视觉特征，512维

        # -----------------UPT提取union feature-----------------
        new_union_features = self.interaction_head(roi_features, obj_feats_vis, proposals, global_features, rel_pair_idxs, self.mode) # 输出1024维
        # alpha = torch.sigmoid(self.alpha)
        # union_features = self.alpha * union_features + (1-self.alpha) * self.up_dim(new_union_features)
        union_features = union_features + self.up_dim(new_union_features)  # 还是这个效果最好

        obj_feats = obj_feats_vis

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            edge_pre_rep_txt = self.obj_embed2(obj_labels)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            edge_pre_rep_txt = self.obj_embed2(obj_preds)

        num_objs = [len(p) for p in proposals]
        tri_reps = []
        obj_ctx_all = obj_feats.split(num_objs)
        obj_label_all = obj_preds.split(num_objs)
        # 三元组分支
        # 分图片来吧 一起真的崩
        for obj_c, obj_l, rel_pair_id, num_obj in zip(obj_ctx_all, obj_label_all, rel_pair_idxs, num_objs):
            num_rel = len(rel_pair_id)
            if num_rel == 0:  # 有这样的图片
                continue
            # tri_rep = torch.cat((obj_c[rel_pair_id[:, 0]], obj_c[rel_pair_id[:, 1]]), dim=-1)
            sub_label = obj_l[rel_pair_id[:, 0]]
            obj_label = obj_l[rel_pair_id[:, 1]]
            sub_embed = self.obj_embed1(sub_label.long())
            obj_embed = self.obj_embed1(obj_label.long())
            tri_rep = self.pred_embed_proj(sub_embed - obj_embed)
            # tri_rep = torch.cat((tri_rep, self.pred_embed_proj(sub_embed - obj_embed)), dim=-1)  # 1024+128 相当于pairwise_tokens
            # tri_rep = self.lin_tri(tri_rep)  # 200→512维
            tri_rep = self.tri_encoder(tri_rep, [num_rel])
            tri_reps.append(tri_rep)
        tri_rep = cat(tri_reps)

        # edge context
        edge_pre_rep_vis = self.lin_edge_visual(edge_pre_rep_vis) # 4608→512
        edge_pre_rep_txt = self.lin_edge_textual(edge_pre_rep_txt) # 200→512
        edge_ctx_vis, _ = self.context_edge(edge_pre_rep_vis, edge_pre_rep_txt, num_objs) # # SHA encoder，只保留第一个返回值，即视觉特征
        edge_ctx = edge_ctx_vis

        return obj_dists, obj_preds, edge_ctx, union_features, tri_rep

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds



if __name__ == '__main__':
    pass


