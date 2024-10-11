import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import MLP
from minvis.video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder_frame
import einops


def build_unified_transformer_decoder(cfg, in_channels, mask_classification=True):
    name = cfg.MODEL.UNIFIED_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


class DepthMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x).sigmoid() if i < self.num_layers - 1 else layer(x)
            # x = F.relu(layer(x))
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class UnifiedTransformerDecoder(VideoMultiScaleMaskedTransformerDecoder_frame):
# class VideoMultiScaleMaskedTransformerDecoder_frame_unified_decoder(VideoMultiScaleMaskedTransformerDecoder_frame):
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            depth_dim: int,
            enforce_input_project: bool,
            num_frames,
            depth_max
    ):
        super().__init__(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            num_frames=num_frames,
        )

        # use 2D positional embedding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.depth_embed = MLP(hidden_dim, hidden_dim, depth_dim, 3)
        self.depth_max = depth_max
        self.query_feat = None
        self.query_embed = None
        self.unified_query_feat = nn.Embedding(num_queries, hidden_dim)
        self.unified_query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_global_feat = nn.Embedding(1, hidden_dim)
        self.query_global_embed = nn.Embedding(1, hidden_dim)

        # feature transform
        self.seg_feat_trans = MLP(256, 256, 256, 2)
        self.depth_feat_trans = MLP(256, 256, 256, 2)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = dict()
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.UNIFIED_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.UNIFIED_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters
        ret["nheads"] = cfg.MODEL.UNIFIED_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.UNIFIED_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.UNIFIED_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.UNIFIED_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.UNIFIED_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.UNIFIED_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["depth_dim"] = cfg.MODEL.UNIFIED_FORMER.DEPTH_DIM
        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM
        ret["depth_max"] = cfg.MODEL.UNIFIED_FORMER.DEPTH_MAX

        return ret

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # unified query
        query_unified_embed = torch.cat([self.query_global_embed.weight, self.unified_query_embed.weight])
        query_unified_feat = torch.cat([self.query_global_feat.weight, self.unified_query_feat.weight])
        query_embed = query_unified_embed.unsqueeze(1).repeat(1, bs, 1)
        output = query_unified_feat.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_depth = []
        predictions_global_depth = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, instance_depth_maps, global_depth, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_depth.append(instance_depth_maps)
        predictions_global_depth.append(global_depth)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # prevent NaN output
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            output = self.transformer_ffn_layers[i](
                output
            )
            outputs_class, outputs_mask, instance_depth_maps, global_depth, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_depth.append(instance_depth_maps)
            predictions_global_depth.append(global_depth)

        assert len(predictions_class) == self.num_layers + 1

        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)
            predictions_depth[i] = einops.rearrange(predictions_depth[i], '(b t) q h w -> b q t h w', t=t)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', t=t)

        pred_embds = self.decoder_norm(output[1:])
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)

        out = {
            "seg_preds": {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'pred_embds': pred_embds,
                'aux_outputs': self._set_aux_loss(
                    predictions_class, predictions_mask, predictions_depth
                ),
            },
            "depth_preds":{
                'pred_depths': predictions_depth[-1],
                'pred_global_depths': predictions_global_depth,
                'aux_outputs': self._set_aux_loss(
                    predictions_class, predictions_mask, predictions_depth
                ),
            },
        }

        return out

    def forward_prediction_heads(self, output, image_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output[:,1:])
        mask_embed = self.mask_embed(decoder_output[:,1:])
        depth_embed = self.depth_embed(decoder_output)

        # feature transform
        seg_features = self.seg_feat_trans(torch.einsum("bchw->bhwc", image_features)).permute(0,3,1,2)
        depth_features = self.depth_feat_trans(torch.einsum("bchw->bhwc", image_features)).permute(0,3,1,2)

        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, seg_features)
        outputs_depth = torch.einsum("bqc,bchw->bqhw", depth_embed, depth_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        scaled_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        global_attn_mask = torch.ones_like(scaled_mask[:,0,:,:]).unsqueeze(1).flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1).bool()
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        instance_attn_mask = (scaled_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = torch.cat([global_attn_mask, instance_attn_mask], dim=1)
        attn_mask = attn_mask.detach()

        norm_depth = outputs_depth.sigmoid()
        final_depth = (norm_depth * (80. - 0.01)) + 0.01
        global_depth = final_depth[:, 0, :, :].unsqueeze(1)
        instance_depth_maps = final_depth[:, 1:, :, :]

        return outputs_class, outputs_mask, instance_depth_maps, global_depth, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_depths):
        return [
            {"pred_logits": a, "pred_masks": b, "pred_depths": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_depths[:-1])
        ]
