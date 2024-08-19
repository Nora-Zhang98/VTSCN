import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import List, Optional, Tuple
from collections import OrderedDict
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
import pocket
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import layer_init

def compute_spatial_encodings(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shape: Tuple[int, int], eps: float = 1e-10
) -> Tensor:
    """
    Parameters:
    -----------
    boxes_1: List[Tensor]
        First set of bounding boxes (M, 4)
    boxes_1: List[Tensor]
        Second set of bounding boxes (M, 4)
    shapes: List[Tuple[int, int]]
        Image shapes, heights followed by widths
    eps: float
        A small constant used for numerical stability

    Returns:
    --------
    Tensor
        Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    # for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
    boxlist1 = BoxList(boxes_1, shape, mode="xyxy")
    boxlist2 = BoxList(boxes_2, shape, mode="xyxy")
    b1 = boxes_1 # tensor
    b2 = boxes_2
    h, w = shape

    c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
    c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

    b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
    b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

    d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
    d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

    iou = torch.diag(boxlist_iou(boxlist1, boxlist2)) # iou

    # Construct spatial encoding
    f = torch.stack([
        # Relative position of box centre
        c1_x / w, c1_y / h, c2_x / w, c2_y / h,
        # Relative box width and height
        b1_w / w, b1_h / h, b2_w / w, b2_h / h,
        # Relative box area
        b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
        b2_w * b2_h / (b1_w * b1_h + eps),
        # Box aspect ratio
        b1_w / (b1_h + eps), b2_w / (b2_h + eps),
        # Intersection over union
        iou,
        # Relative distance and direction of the object w.r.t. the person
        (c2_x > c1_x).float() * d_x,
        (c2_x < c1_x).float() * d_x,
        (c2_y > c1_y).float() * d_y,
        (c2_y < c1_y).float() * d_y,
    ], 1)

    features.append(
        torch.cat([f, torch.log(f + eps)], 1)
    )
    return torch.cat(features)

class MultiBranchFusion(nn.Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    hidden_state_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        hidden_state_size: int, cardinality: int
    ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(hidden_state_size / cardinality)
        assert sub_repr_size * cardinality == hidden_state_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, hidden_state_size)
            for _ in range(cardinality)
        ])
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))

class ModifiedEncoderLayer(nn.Module):
    def __init__(self,
        hidden_size: int = 256, representation_size: int = 512,
        num_heads: int = 8, dropout_prob: float = .1, return_weights: bool = False,
    ) -> None:
        super().__init__()
        if representation_size % num_heads != 0:
            raise ValueError(
                f"The given representation size {representation_size} "
                f"should be divisible by the number of attention heads {num_heads}."
            )
        self.sub_repr_size = int(representation_size / num_heads)

        self.hidden_size = hidden_size
        self.representation_size = representation_size

        self.num_heads = num_heads
        self.return_weights = return_weights

        self.unary = nn.Linear(hidden_size*2, representation_size)
        self.pairwise = nn.Linear(representation_size, representation_size)
        self.attn = nn.ModuleList([nn.Linear(3 * self.sub_repr_size, 1) for _ in range(num_heads)])
        self.message = nn.ModuleList([nn.Linear(self.sub_repr_size, self.sub_repr_size) for _ in range(num_heads)])
        self.aggregate = nn.Linear(representation_size, hidden_size*2)
        self.norm = nn.LayerNorm(hidden_size*2)
        self.dropout = nn.Dropout(dropout_prob)
        self.ffn = pocket.models.FeedForwardNetwork(hidden_size*2, hidden_size * 4, dropout_prob)

    def reshape(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.sub_repr_size
        )
        x = x.view(*new_x_shape)
        if len(new_x_shape) == 3:
            return x.permute(1, 0, 2)
        elif len(new_x_shape) == 4:
            return x.permute(2, 0, 1, 3)
        else:
            raise ValueError("Incorrect tensor shape")

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        device = x.device
        n = len(x)

        u = F.relu(self.unary(x)) 
        p = F.relu(self.pairwise(y)) 

        # Unary features (H, N, L)
        u_r = self.reshape(u)
        # Pairwise features (H, N, N, L)
        p_r = self.reshape(p)

        i, j = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )

        # Features used to compute attention (H, N, N, 3L)
        attn_features = torch.cat([
            u_r[:, i], u_r[:, j], p_r
        ], dim=-1) 
        # Attention weights (H,) (N, N, 1)
        weights = [
            F.softmax(l(f), dim=0) for f, l
            in zip(attn_features, self.attn)
        ] 
        # Repeated unary feaures along the third dimension (H, N, N, L)
        u_r_repeat = u_r.unsqueeze(dim=2).repeat(1, 1, n, 1) 
        messages = [
            l(f_1 * f_2) for f_1, f_2, l
            in zip(u_r_repeat, p_r, self.message)
        ]

        aggregated_messages = self.aggregate(F.relu(
            torch.cat([
                (w * m).sum(dim=0) for w, m
                in zip(weights, messages)
            ], dim=-1) 
        ))
        aggregated_messages = self.dropout(aggregated_messages)
        x = self.norm(x + aggregated_messages) 
        x = self.ffn(x)

        if self.return_weights: attn = weights
        else: attn = None

        return x, attn


class ModifiedEncoder(nn.Module):
    def __init__(self,
                 hidden_size: int = 256, representation_size: int = 512,
                 num_heads: int = 8, num_layers: int = 2,
                 dropout_prob: float = .1, return_weights: bool = False
                 ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.mod_enc = nn.ModuleList([ModifiedEncoderLayer(
            hidden_size=hidden_size, representation_size=representation_size,
            num_heads=num_heads, dropout_prob=dropout_prob, return_weights=return_weights
        ) for _ in range(num_layers)])

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, List[Optional[Tensor]]]:
        attn_weights = []
        for layer in self.mod_enc:
            x, attn = layer(x, y)
            attn_weights.append(attn)
        return x, attn_weights

class InteractionHead(nn.Module):
    """
    Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_pair_predictor: nn.Module
        Module that classifies box pairs
    hidden_state_size: int
        Size of the object features
    representation_size: int
        Size of the human-object pair features
    num_channels: int
        Number of channels in the global image features
    num_classes: int
        Number of target classes
    human_idx: int
        The index of human/person class
    object_class_to_target_class: List[list]
        The set of valid action classes for each object type
    """
    def __init__(self,
        hidden_state_size: int, representation_size: int,
        num_channels: int
    ) -> None:
        super().__init__()

        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, representation_size),
            nn.ReLU(),
        )

        self.coop_layer = ModifiedEncoder(
            hidden_size=hidden_state_size,
            representation_size=representation_size,
            num_layers=2, 
            return_weights=True
        )
        self.coop_layer_dis = ModifiedEncoder(
            hidden_size=hidden_state_size,
            representation_size=representation_size,
            num_layers=1,
            return_weights=True, 
        )
        self.coop_layer_int = ModifiedEncoder(
            hidden_size=hidden_state_size,
            representation_size=representation_size,
            num_layers=1,
            return_weights=True  
        )
        self.comp_layer = pocket.models.TransformerEncoderLayer(
            hidden_size=representation_size * 2,
            return_weights=True
        )

        self.mbf = MultiBranchFusion(
            hidden_state_size * 2,
            representation_size, representation_size,
            cardinality=16
        )

        self.mbf_g = MultiBranchFusion(
            1280, representation_size,
            representation_size, cardinality=16
        )

        self.reduce_channel = nn.Linear(hidden_state_size*4, hidden_state_size*2)
        self.pairwise_embed = nn.Linear(1024+200, 1024)
        layer_init(self.pairwise_embed, xavier=True)


    def forward(self, features, obj_ctx, proposals, global_features, rel_pair_idxs, mode):
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        image_shapes: Tensor
            (B, 2) Image shapes, heights followed by widths
        region_props: List[dict]
            Region proposals with the following keys
            `boxes`: Tensor
                (N, 4) Bounding boxes
            `scores`: Tensor
                (N,) Object confidence scores
            `labels`: Tensor
                (N,) Object class indices
            `hidden_states`: Tensor
                (N, 256) Object features
        """
        
        device = features.device
        boxes_per_image = [len(prp.bbox) for prp in proposals]
        obj_ctxs = obj_ctx.split(boxes_per_image, dim=0)
        pairwise_tokens_collated = []

        for proposal, obj_ctx, global_feature, rel_pair_id in zip(proposals, obj_ctxs, global_features, rel_pair_idxs): 
            box = proposal.bbox
            rel_pair_id = rel_pair_id.long()
            n = len(box)
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x = x.flatten(); y = y.flatten()
            if mode == 'sgdet':
                x_keep, y_keep = rel_pair_id[:, 0], rel_pair_id[:, 1]  

                box_pair_spatial = compute_spatial_encodings(box[x_keep], box[y_keep], proposal.size)  
                box_pair_spatial = self.spatial_head(box_pair_spatial)
                box_pair_spatial_reshaped = torch.zeros(n, n, 512).to(device)
                box_pair_spatial_reshaped[x_keep, y_keep] = box_pair_spatial

                unary_tokens, unary_attn = self.coop_layer_int(obj_ctx, box_pair_spatial_reshaped)
            else:
                box_pair_spatial_all = compute_spatial_encodings(box[x], box[y], proposal.size) 
                box_pair_spatial_all = self.spatial_head(box_pair_spatial_all)  
                box_pair_spatial_reshaped_all = box_pair_spatial_all.reshape(n, n, -1)

                unary_tokens_dis, unary_attn_dis = self.coop_layer_dis(obj_ctx, box_pair_spatial_reshaped_all)

                x_keep, y_keep = rel_pair_id[:,0], rel_pair_id[:,1]
   
                box_pair_spatial = compute_spatial_encodings(box[x_keep], box[y_keep], proposal.size)
                box_pair_spatial = self.spatial_head(box_pair_spatial) 
                box_pair_spatial_reshaped = torch.zeros(n, n, 512).to(device)
                box_pair_spatial_reshaped[x_keep, y_keep] = box_pair_spatial

                unary_tokens, unary_attn = self.coop_layer_int(unary_tokens_dis, box_pair_spatial_reshaped)

            pairwise_tokens = torch.cat([
                self.mbf(  
                    self.reduce_channel(torch.cat([unary_tokens[x_keep], unary_tokens[y_keep]], 1)),
                    box_pair_spatial_reshaped[x_keep, y_keep]
                ), self.mbf_g(  
                    global_feature[None, :],  
                    box_pair_spatial_reshaped[x_keep, y_keep]),
            ], dim=1)

            # Run the competitive layer# !-----------Multi Branch Fusion------------
            pairwise_tokens, pairwise_attn = self.comp_layer(pairwise_tokens) 
         
            pairwise_tokens_collated.append(pairwise_tokens)

        pairwise_tokens_collated = torch.cat(pairwise_tokens_collated)
        return pairwise_tokens_collated
