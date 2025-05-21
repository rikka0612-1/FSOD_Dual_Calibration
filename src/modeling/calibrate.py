import logging
import pickle as pkl
from collections import defaultdict
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, configurable
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_backbone
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.structures import ImageList
from einops import repeat

logger = logging.getLogger("detectron2")


@lru_cache(maxsize=None)
def extract_support_rois(cfg_bytes: bytes):
    # Deserialize config
    cfg = pkl.loads(cfg_bytes)
    device = torch.device(cfg.MODEL.DEVICE)

    # Initialize backbone and pooler
    cfg.defrost()
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res5"]
    cfg.freeze()
    backbone = build_backbone(cfg)
    backbone.stem.fc = nn.Linear(2048, 1000)
    DetectionCheckpointer(backbone).resume_or_load("./pretrain/R-101.pkl")
    backbone.eval()
    backbone.to(device)
    for param in backbone.parameters():
        param.requires_grad_(False)
    pooler = ROIPooler(
        output_size=(1, 1),
        scales=(1.0 / 32,),
        sampling_ratio=0,
        pooler_type="ROIAlignV2"
    )

    # image normalization
    mean = torch.tensor(cfg.MODEL.PIXEL_MEAN, device=device).view(-1, 1, 1)
    std = torch.tensor(cfg.MODEL.PIXEL_STD, device=device).view(-1, 1, 1)

    # collect all support samples
    support_samples = []
    for dataset_name in cfg.DATASETS.TRAIN:
        dataset = DatasetCatalog.get(dataset_name)
        support_samples.extend(dataset)

    # Collect features for each category
    category_features = {}
    for sample in support_samples:
        # Read and preprocess image
        img = utils.read_image(sample["file_name"], format=cfg.INPUT.FORMAT)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
        img_tensor = (img_tensor - mean) / std
        imglist = ImageList.from_tensors([img_tensor], 0) 
        # Extract features
        with torch.no_grad():
            feat_map = backbone(imglist.tensor)["res5"]
        # Get instances and ROI features
        instances = utils.annotations_to_instances(sample["annotations"], img_tensor.shape[1:])
        boxes = instances.gt_boxes.to(device)
        pooled_feats = pooler([feat_map], [boxes]).flatten(1)
        for i, cls_id in enumerate(instances.gt_classes.tolist()):
            if cls_id not in category_features:
                category_features[cls_id] = []
            category_features[cls_id].append(pooled_feats[i])

    # Assemble support features and masks
    shot = int(cfg.DATASETS.TRAIN[0].split("_")[4].replace("shot", ""))
    all_feats, all_masks = [], []
    for cls in sorted(category_features.keys()):
        feats = category_features[cls]
        mask = [1] * min(len(feats), shot)
        if len(feats) < shot:
            # Pad zeros if not enough shots
            pad = [torch.zeros_like(feats[0]) for _ in range(shot - len(feats))]
            feats = feats + pad
            mask += [0] * (shot - len(mask))
        else:
            feats = feats[:shot]
        all_feats.append(torch.stack(feats))
        all_masks.append(torch.tensor(mask, dtype=torch.float32, device=feats[0].device).unsqueeze(1))

    support_tensor = torch.stack(all_feats).cpu()
    mask_tensor = torch.stack(all_masks).cpu()
    return support_tensor, mask_tensor


class Calibrate(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        dropout: nn.Module = nn.Identity(),
        supp_features: torch.Tensor = None,
        transform_func: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__(input_shape, **kwargs)
        self.dropout = dropout
        self.register_buffer("fsup", supp_features)
        self.fc = transform_func

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.zeros_(self.bbox_pred.bias)
        nn.init.zeros_(self.cls_score.bias)

    @classmethod
    def from_config(cls, cfg: CfgNode, input_shape: "ShapeSpec"):
        ret = super().from_config(cfg, input_shape)

        supp_feats, mask = extract_support_rois(pkl.dumps(cfg))
        ret["supp_features"] = torch.sum(supp_feats * mask, dim=1) / mask.sum(dim=1)

        fc = cls.learn_mapping(supp_feats, mask, cfg.MODEL.DEVICE)
        ret["transform_func"] = fc

        if cfg.MODEL.ROI_HEADS.DROPOUT > 0:
            drop_ratio = cfg.MODEL.ROI_HEADS.DROPOUT
            ret["dropout"] = nn.Dropout(p=drop_ratio)
            logger.info((f"[CLS] Use dropout: p = {ret['dropout'].p}"))

        return ret

    @classmethod
    @lru_cache(maxsize=None)
    def learn_mapping(
        cls, support_feats, mask, device_id: str = "cpu", tol: float = 1e-6, max_iter: int = 1000
    ):

        device = torch.device(device_id)
        n_cls, n_shot, feat_dim = support_feats.shape

        # Flatten features and mask
        flat_feats = support_feats.view(-1, feat_dim).to(device)
        flat_mask = mask.view(-1, 1).to(device)
        mask_matrix = flat_mask @ flat_mask.t()

        # Build target similarity matrix
        labels = torch.arange(n_cls, device=device).repeat_interleave(n_shot)
        target = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # Initialize linear mapping
        mapper = nn.Linear(feat_dim, feat_dim, bias=False).to(device)
        nn.init.normal_(mapper.weight, mean=0, std=0.01)
        optimizer = torch.optim.SGD(mapper.parameters(), lr=1.0)

        last_loss = None
        for step in range(max_iter):
            proj = mapper(flat_feats)
            sim = F.normalize(proj, dim=1) @ F.normalize(proj, dim=1).t()
            mse = (sim - target) ** 2
            loss = (mse * mask_matrix).sum() / mask_matrix.sum()

            if last_loss is not None and abs(loss.item() - last_loss) < tol:
                break
            last_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"[learn_mapping-re] Final loss: {loss.item():.6f}")
        return mapper.cpu()

    @staticmethod
    def cosine_similarity(x, y, eps=1e-8):
        return F.normalize(x, dim=-1, eps=eps).matmul(F.normalize(y, dim=-1, eps=eps).T)

    def forward(self, x):
        # Flatten features if needed
        if x.dim() > 2:
            features = x.flatten(start_dim=1)
        else:
            features = x

        # Bounding box regression branch
        bbox_deltas = self.bbox_pred(features)

        # Classification branch
        logits = self.cls_score(self.dropout(features))

        # Feature space mapping
        support_proj = self.fc(self.fsup)
        query_proj = self.fc(features)
        sim_scores = self.cosine_similarity(query_proj, support_proj)

        # Normalization factors
        with torch.no_grad():
            norm_logits = logits.norm(dim=1, keepdim=True)
            norm_sim = sim_scores.norm(dim=1, keepdim=True)

        # Calibrate logits and similarity scores
        logits_calibrated = logits * norm_sim
        sim_calibrated = sim_scores * norm_logits

        
        num_fg = sim_scores.shape[1]
        fg_logits, bg_logits = logits_calibrated.split(num_fg, dim=1)
        final_scores = torch.cat([fg_logits + sim_calibrated, bg_logits + norm_logits], dim=1) / 2

        return final_scores, bbox_deltas
