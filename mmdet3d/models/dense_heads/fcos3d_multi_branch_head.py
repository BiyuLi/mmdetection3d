from mmdet3d.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import build_bbox_coder
from mmcv.cnn import ConvModule, Scale, normal_init
from mmdet.core import multi_apply
from mmdet3d.core import (box3d_multiclass_nms, limit_period, points_img2cam,
                          xywhr2xyxyr)
from mmcv.runner import BaseModule
import torch
import torch.nn as nn
import numpy as np

INF = 1e8
@HEADS.register_module()
class HppFCOSOneStageMono3DHead(BaseModule):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs,
                 feat_channels,
                 num_direction_classfier,
                 diff_rad_by_sin,
                 strides,
                 group_reg_dims,
                 group_reg_dims_3d,
                 cls_branch,
                 reg_branch,
                 dir_branch,
                 centerness_branch,
                 centerness3d_branch,
                 loss_cls,
                 loss_iou,
                 loss_bbox,
                 loss_dir,
                 loss_centerness,
                 loss_centerness3d,
                 loss_rot,
                 bbox_coder,
                 bbox_code_size,
                 norm_on_bbox,
                 centerness_on_reg,
                 centerness_alpha,
                 center_sampling,
                 center_sample_radius,
                 dir_offset,
                 dir_limit_offset,
                 conv_bias,
                 background_label,
                 regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384),
                                 (384, INF)),
                 train_cfg=None,
                 test_cfg=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(HppFCOSOneStageMono3DHead, self).__init__()
        self.regress_ranges = regress_ranges
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.num_direction_classifier = num_direction_classfier
        self.diff_rad_by_sin = diff_rad_by_sin
        self.strides = strides
        assert len(reg_branch) == len(group_reg_dims), 'The number' \
        'of element in reg_branch and group_reg_dims should be the same'
        self.group_reg_dims = group_reg_dims
        self.group_reg_dims_3d = group_reg_dims_3d
        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        self.dir_branch = dir_branch
        self.centerness_branch = centerness_branch
        self.centerness3d_branch = centerness3d_branch
        self.loss_cls = build_loss(loss_cls)
        self.loss_iou = build_loss(loss_iou)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_centerness3d = build_loss(loss_centerness3d)
        self.loss_rot = build_loss(loss_rot)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.bbox_code_size = bbox_code_size
        self.norm_on_box = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.centerness_alpha = centerness_alpha
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg
        self.conv_bias = conv_bias
        self.out_channels = []
        for reg_branch_channels in reg_branch:
            if len(reg_branch_channels) > 0:
                self.out_channels.append(reg_branch_channels[-1])
            else:
                self.out_channels.append(-1)
        self.fp16_enabled = False
        self.background_label = (num_classes if background_label is None else background_label)
        assert self.background_label == 0 or self.background_label == num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.scale_dim = 3
        self.scales = nn.ModuleList([
            nn.ModuleList([Scale(1.0) for _ in range(self.scale_dim)])
            for _ in self.strides
        ])
        self.person_cls_branch = self.build_cls_layer(self.cls_branch, (1,) * len(self.cls_branch))
        self.person_reg_branch = self.build_regession_layer(self.group_reg_dims_3d)
        self.vehicle_cls_branch = self.build_cls_layer(self.cls_branch, (1,) * len(self.cls_branch))
        self.vehicle_reg_branch = self.build_regession_layer(self.group_reg_dims_3d)
        self.rider_cls_branch = self.build_cls_layer(self.cls_branch, (1,) * len(self.cls_branch))
        self.rider_reg_branch = self.build_regession_layer(self.group_reg_dims_3d)
        self.rear_cls_branch = self.build_cls_layer(self.cls_branch, (1,) * len(self.cls_branch))
        self.rear_reg_branch = self.build_regession_layer(self.group_reg_dims)

    def build_cls_layer(self,conv_channels=(64), conv_strides=(1)):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    padding=1,
                    stride=1,
                    bias=self.conv_bias
                )
            )
        conv_before_cls_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)

        for i in range(len(conv_strides)):
            conv_before_cls_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    bias=self.conv_bias
                )
            )
        cls_pred = nn.Conv2d(conv_channels[-1], 1, 1)

        return nn.Sequential(*cls_convs, *conv_before_cls_pred, cls_pred)

    def build_regession_layer(self, regression_dim_group):
        total_structure = nn.ModuleList()
        reg_convs = nn.Sequential()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=self.conv_bias
                )
            )

        """
        (bbox, offset, depth, dim, head_dir, rot, centerness, centerness3d)or 
        (4, 2, 1, 3, 4, 4, 1, 1)
        (bbox, centerness) (4, 1)
        """
        for out_channel in regression_dim_group:
            temp_conv = nn.Sequential()
            temp_conv.append(
                    ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        bias=self.conv_bias
                    ))
            temp_conv.append(nn.Conv2d(
                        self.feat_channels,
                        out_channel,
                        1,
                    )
                )
            total_structure.append(temp_conv)


        return [reg_convs, total_structure]

    def _forward_classification(self, cls_pred, cls_feat):
        for block in cls_pred:
            cls_feat = block(cls_feat)
        return cls_feat

    def _forward_regression(self, reg_pred, reg_feat):
        feature_extraction = reg_pred[0]
        reg_feat = feature_extraction(reg_feat)
        res_list = []
        for attr in reg_pred[1]:
            feature_map = reg_feat.clone()
            feature_map = attr(feature_map)
            res_list.append(feature_map)

        return res_list

    def init_weights(self):
        for module in [self.person_cls_branch, self.person_reg_branch ,self.vehicle_cls_branch,self.vehicle_reg_branch,\
            self.rider_cls_branch, self.rider_reg_branch, self.rear_cls_branch, self.rear_reg_branch]:
            for m in module:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

    def forward_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x

        #person branch
        """
        person channel:
            input featmap size : (2, 64, 128, 128) bs = 2, feat channel = 64 
            output size:
            torch.Size([2, 1, 128, 128]) cls
            torch.Size([2, 4, 128, 128]) 2dbox
            torch.Size([2, 2, 128, 128]) offset
            torch.Size([2, 1, 128, 128]) depth
            torch.Size([2, 3, 128, 128]) size
            torch.Size([2, 4, 128, 128]) dir_offset
            torch.Size([2, 4, 128, 128]) dir_cls
            torch.Size([2, 1, 128, 128]) centerness2d
            torch.Size([2, 1, 128, 128]) centerness3d
        """
        person_cls_feat = cls_feat.clone()
        person_cls_pred = self._forward_classification(self.person_cls_branch, person_cls_feat)

        person_reg_feat = reg_feat.clone()
        [person_bbox, person_offset, person_depth, person_dim, person_dir_cls,\
        person_dir_offset, person_centerness, person_centerness3d] = \
            self._forward_regression(self.person_reg_branch, person_reg_feat)

#         person_offset, person_depth, person_dim = self.bbox_coder.decode(person_offset, person_depth, person_dim, scale,
#                                                                          stride,self.training, person_cls_pred)

        person_result = [person_cls_pred,person_bbox, person_offset, person_depth, person_dim, person_dir_cls,\
                        person_dir_offset, person_centerness, person_centerness3d]

        #vehicle branch
        vehicle_cls_feat = cls_feat.clone()
        vehicle_cls_pred = self._forward_classification(self.vehicle_cls_branch, vehicle_cls_feat)

        vehicle_reg_feat = reg_feat.clone()
        [vehicle_bbox, vehicle_offset, vehicle_depth, vehicle_dim, vehicle_dir_cls,\
        vehicle_dir_offset,vehicle_centerness, vehicle_centerness3d] = \
            self._forward_regression(self.vehicle_reg_branch, vehicle_reg_feat)

#         vehicle_offset, vehicle_depth, vehicle_dim = self.bbox_coder.decode(vehicle_offset, vehicle_depth, vehicle_dim,
#                                                                         scale,stride,self.training, vehicle_cls_pred)

        vehicle_result = [vehicle_cls_pred,vehicle_bbox, vehicle_offset, vehicle_depth, vehicle_dim, vehicle_dir_cls,\
        vehicle_dir_offset, vehicle_centerness, vehicle_centerness3d]

        #rider branch
        rider_cls_feat = cls_feat.clone()
        rider_cls_pred = self._forward_classification(self.rider_cls_branch, rider_cls_feat)

        rider_reg_feat = reg_feat.clone()
        [rider_bbox, rider_offset, rider_depth, rider_dim, rider_dir_cls,\
        rider_dir_offset,rider_centerness, rider_centerness3d] = \
            self._forward_regression(self.rider_reg_branch, rider_reg_feat)

#         rider_offset, rider_depth, rider_dim = self.bbox_coder.decode(rider_offset, rider_depth, rider_dim, scale,
#                                                                          stride,self.training, rider_cls_pred)

        rider_result = [rider_cls_pred,rider_bbox, rider_offset, rider_depth, rider_dim, rider_dir_cls,\
        rider_dir_offset, rider_centerness, rider_centerness3d]

        #rear branch
        rear_cls_feat = cls_feat.clone()
        rear_cls_pred = self._forward_classification(self.rear_cls_branch, rear_cls_feat)

        rear_reg_feat = reg_feat.clone()
        [rear_bbox, rear_centerness] = self._forward_regression(self.rear_reg_branch, rear_reg_feat)

        rear_result = [rear_cls_pred, rear_bbox, rear_centerness]

        return person_result, vehicle_result, rider_result, rear_result

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    @staticmethod
    def add_sin_difference(rot1, rot2):
        """
        Convert the rotation difference to difference in sine function
        :param rot1: ori rot
        :param rot2: gt rot
        :return: sin encode rot
        """
        #sin(2(a(pred) - b(gt))) = sinacosb - cosasinb  0, pi/2, pi  %pi/2  0-pi ==> 0-pi/2
        # a - b
        rot1, rot2 = rot1 * 2, rot2 * 2
        rad_pred_encoding = torch.sin(rot1) * torch.cos(rot2)
        rad_gt_encoding = torch.cos(rot1) * torch.sin(rot2)

        return rad_pred_encoding, rad_gt_encoding

    @staticmethod
    def get_direction_target(rot_targets,
                             dir_offset=0,
                             dir_limit_offset=0.0,
                             num_bins=4,
                             one_hot=True):
        """
        Encode direction to 0 - num_bins -1
        :param rot_targets:  Rot regression target
        :param dir_offset: Default to 0
        :param dir_limit_offset: Offset to set the direction range. default to 0
        :param num_bins: Number of bins to divide 2*PI. Default to 4
        :param one_hot: Whether to encode as one hot, bool
        :return: Encoded direction targets.
        """
        offset_rot = limit_period(rot_targets - dir_offset, dir_limit_offset, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(
                *list(dir_cls_targets.shape),
                num_bins,
                dtype=rot_targets.dtype,
                device=dir_cls_targets.device)
            dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    #TODO force_fp32

    def loss(self,
             person_preds,
             vehicle_preds,
             rider_preds,
             rear_preds,
             person_gt,
             vehicle_gt,
             rider_gt,
             rear_gt,
             img_metas):
        """Compute loss of the head"""
        #TODO convert gt into 4 branch gt
        featmap_sizes = [featmap[0].size()[-2:] for featmap in person_preds]
        all_level_points = self.get_points(featmap_sizes, person_preds[0][0].dtype, person_preds[0][0].device)
        person_gt, vehicle_gt, rider_gt, rear_gt = self.get_targets(all_level_points, person_gt, vehicle_gt,
                                                                        rider_gt, rear_gt)
        total_loss_cls = 0
        total_loss_bbox = 0
        total_loss_offset = 0
        total_loss_depth = 0
        total_loss_dim = 0
        total_loss_dir_cls = 0
        total_loss_dir_offset = 0
        total_loss_centerness = 0
        total_loss_centerness3d = 0

        #process 3d branch
        for prediction, gt in zip([person_preds, vehicle_preds, rider_preds], [person_gt, vehicle_gt, rider_gt]):
#             [cls_pred, bbox, offset, depth, dim, dir_cls, dir_offset, centerness, centerness3d] = prediction
            cls_pred, bbox, offset, depth, dim, dir_cls, dir_offset, centerness, centerness3d = [],[],[],[],[],[],[],[],[]
            for i in range(5):
                cls_pred.append(prediction[i][0])
                bbox.append(prediction[i][1])
                offset.append(prediction[i][2])
                depth.append(prediction[i][3])
                dim.append(prediction[i][4])
                dir_cls.append(prediction[i][5])
                dir_offset.append(prediction[i][6])
                centerness.append(prediction[i][7])
                centerness3d.append(prediction[i][8])
            [labels, bbox_target, bbox_targets_3d, centerness2d_targets, centerness3d_targets] = gt
            num_imgs = cls_pred[0].size(0)
            
            flatten_cls_scores = [
                cls_score.permute(0, 2, 3, 1).reshape(-1, 1) for cls_score in cls_pred
            ]
            flatten_bbox_preds = [
                bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox
            ]
            flatten_offset_preds = [
                offset_pred.permute(0, 2, 3, 1).reshape(-1, 2) for offset_pred in offset
            ]
            flatten_depth_preds = [
                depth_pred.permute(0, 2, 3, 1).reshape(-1) for depth_pred in depth
            ]
            flatten_dim_preds = [
                dim_pred.permute(0, 2, 3, 1).reshape(-1, 3) for dim_pred in dim
            ]
            flatten_dir_cls_preds = [
                dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 4) for dir_cls_pred in dir_cls
            ]
            flatten_dir_offset_preds = [
                dir_offset_pred.permute(0, 2, 3, 1).reshape(-1, 4) for dir_offset_pred in dir_offset
            ]
            flatten_centerness_preds = [
                centerness_pred.permute(0, 2, 3, 1).reshape(-1) for centerness_pred in centerness
            ]
            flatten_centerness3d_preds = [
                centerness3d_pred.permute(0, 2, 3, 1).reshape(-1) for centerness3d_pred in centerness3d
            ]

            #process predictions
            flatten_cls_scores = torch.cat(flatten_cls_scores)
            flatten_bbox_preds = torch.cat(flatten_bbox_preds)
            flatten_offset_preds = torch.cat(flatten_offset_preds)
            flatten_depth_preds = torch.cat(flatten_depth_preds)
            flatten_dim_preds = torch.cat(flatten_dim_preds)
            flatten_dir_cls_preds = torch.cat(flatten_dir_cls_preds)
            flatten_dir_offset_preds = torch.cat(flatten_dir_offset_preds)
            flatten_centerness_preds = torch.cat(flatten_centerness_preds)
            flatten_centerness3d_preds = torch.cat(flatten_centerness3d_preds)

            #process target
            flatten_label = torch.cat(labels)
            flatten_bbox_target = torch.cat(bbox_target)
            flatten_bbox3d_target = torch.cat(bbox_targets_3d)
            flatten_centerness_target = torch.cat(centerness2d_targets)
            flatten_centerness3d_target = torch.cat(centerness3d_targets)

            pos_inds = (flatten_label == 1).nonzero().reshape(-1)
            num_pos = len(pos_inds)
            
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_label,
                avg_factor = num_pos + num_imgs
            )

            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_offset_preds = flatten_offset_preds[pos_inds]
            pos_depth_preds = flatten_depth_preds[pos_inds]
            pos_dim_preds = flatten_dim_preds[pos_inds]
            pos_dir_cls_preds = flatten_dir_cls_preds[pos_inds]
            pos_dir_offset_preds = flatten_dir_offset_preds[pos_inds]
            pos_centerness_preds = flatten_centerness_preds[pos_inds]
            pos_centerness3d_preds = flatten_centerness3d_preds[pos_inds]

            if num_pos > 0:
                pos_bbox_target = flatten_bbox_target[pos_inds]
                pos_offset_target = flatten_bbox3d_target[pos_inds][0:2]
                pos_depth_target = flatten_bbox3d_target[pos_inds][2]
                pos_dim_target = flatten_bbox3d_target[pos_inds][3:6]
                pos_rot_target = flatten_bbox3d_target[pos_inds][6]
                pos_dir_cls_target = self.get_direction_target(
                    pos_rot_target,
                    self.dir_offset,
                    self.dir_limit_offset,
                    one_hot=False
                )
                pos_centerness_target = flatten_centerness_target[pos_inds]
                pos_centerness3d_target = flatten_centerness3d_target[pos_inds]

                bbox_weights = pos_centerness_target.new_ones(len(pos_centerness_target), sum(self.group_reg_dims))
                equal_weights = pos_centerness_target.new_ones(pos_centerness_target.shape)
                code_weight = self.train_cfg.get('code_weight', None)  #TODO need to check
                if code_weight:
                    assert len(code_weight) == sum(self.group_reg_dims)
                    bbox_weights = bbox_weights * bbox_weights.new_tensor(code_weight)

                if self.diff_rad_by_sin:
                    pos_dir_offset_preds, pos_rot_target = self.add_sin_difference(pos_dir_offset_preds, pos_rot_target) #TODO need to check the channel of dir_offset
                
                loss_bbox = self.loss_iou(pos_bbox_preds, pos_bbox_target)
                loss_offset = self.loss_bbox(pos_offset_preds, pos_offset_target, weight=bbox_weights[:, :2], avg_factor=equal_weights.sum())
                loss_depth = self.loss_bbox(pos_depth_preds, pos_depth_target, weight=bbox_weights[:, 2],  avg_factor=equal_weights.sum())
                loss_dim = self.loss_bbox(pos_dim_preds, pos_dim_target, weight=bbox_weights[:, 3:6], avg_factor=equal_weights.sum())
                loss_rot_sin = self.loss_bbox(pos_dir_offset_preds, pos_rot_target, weight=bbox_weights[:, 6], avg_factor=equal_weights.sum())
                loss_centerness = self.loss_centerness(pos_centerness_preds, pos_centerness_target)
                loss_centerness3d = self.loss_centerness3d(pos_centerness3d_preds, pos_centerness3d_target)
                loss_dir = self.loss_dir(pos_dir_cls_preds, pos_dir_cls_target, equal_weights, avg_factor=equal_weights.sum())
            else:
                loss_bbox = pos_bbox_preds.sum()
                loss_offset = pos_offset_preds.sum()
                loss_depth = pos_depth_preds.sum()
                loss_dim = pos_dim_preds.sum()
                loss_rot_sin = pos_dir_offset_preds.sum()
                loss_dir = pos_dir_cls_preds.sum()
                loss_centerness = pos_centerness_preds.sum()
                loss_centerness3d = pos_centerness3d_preds.sum()
            
            #TODO different weight with different branch
            total_loss_cls += loss_cls
            total_loss_bbox += loss_bbox
            total_loss_depth += loss_depth
            total_loss_dim += loss_dim
            total_loss_offset += loss_offset
            total_loss_dir_offset += loss_rot_sin
            total_loss_dir_cls += loss_dir
            total_loss_centerness += loss_centerness
            total_loss_centerness3d += loss_centerness3d
        
        #process rear
        predction, gt = rear_preds, rear_gt
        
#         cls_pred, bbox, centerness = gt
        labels, bbox_target, centerness2d_targets = gt
        num_imgs = cls_pred[0].size(0)
        cls_pred, bbox, centerness = [],[],[]
        for i in range(5):
            cls_pred.append(prediction[i][0])
            bbox.append(prediction[i][1])
            centerness.append(prediction[i][2])

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, 1) for cls_score in cls_pred
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox
        ]
        flatten_centerness_preds = [
            centerness_pred.permute(0, 2, 3, 1).reshape(-1) for centerness_pred in centerness
        ]


        #process predictions
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness_preds = torch.cat(flatten_centerness_preds)

        #process target
        flatten_label = torch.cat(labels)
        flatten_bbox_target = torch.cat(bbox_target)
        flatten_centerness_target = torch.cat(centerness2d_targets)


        pos_inds = (flatten_label == 1).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_label,
            avg_factor = num_pos + num_imgs
        )

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness_preds = flatten_centerness_preds[pos_inds]

        if num_pos > 0:
            pos_bbox_target = flatten_bbox_target[pos_inds]
            pos_centerness_target = flatten_centerness_target[pos_inds]

            loss_bbox = self.loss_iou(pos_bbox_preds, pos_bbox_target)
            loss_centerness = self.loss_centerness(pos_centerness_preds, pos_centerness_target)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness_preds.sum()

        total_loss_cls += loss_cls
        total_loss_bbox += loss_bbox
        total_loss_centerness += loss_centerness

        loss_dict = dict(
            loss_cls=total_loss_cls,
            loss_bboxes=total_loss_bbox,
            loss_offset=total_loss_offset,
            loss_depth=total_loss_depth,
            loss_dim=total_loss_dim,
            loss_dir=total_loss_dir_cls,
            loss_rotsin=total_loss_dir_offset,
            loss_centerness=total_loss_centerness,
            loss_centerness3d=total_loss_centerness3d
        )

        return loss_dict
        

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points

    def _get_points_single(self, featmap_size, stride, dtype, device, flatten=False):
        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)

        if flatten:
            y = y.flatten()
            x = x.flatten()
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1) + stride // 2

        return points

    def get_targets(self, points, person_gt, vehicle_gt, rider_gt, rear_gt):
        """
        person_gt(List): [bbox2d, bbox_cam3d, center2ds]
                            bbox2d[x, y, x, y]
                            bbox_cam3d[x, y, z, w, h, l, yaw]
                            centers2d[x, y, depth]
        """
        assert len(points) == len(self.regress_ranges)
        num_points = [center.size(0) for center in points]
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        [person_bboxes, person_bboxes_3d, person_centers2d] = person_gt
        [vehicle_bboxes, vehicle_bboxes_3d, vehicle_centers2d] = vehicle_gt
        [rider_bboxes, rider_bboxes_3d, rider_centers2d] = rider_gt
        rear_bboxes = rear_gt
        person_labels, person_bbox_targets, person_bbox_targets_3d, person_centerness2d_targets, person_centerness3d_targets,\
            vehicle_labels, vehicle_bbox_targets, vehicle_bbox_targets_3d, vehicle_centerness2d_targets, vehicle_centerness3d_targets,\
            rider_labels, rider_bbox_targets, rider_bbox_targets_3d, rider_centerness2d_targets, rider_centerness3d_targets,\
            rear_labels, rear_bbox_targets, rear_centerness2d_targets = multi_apply(
                self._get_target_single,
                person_bboxes, 
                person_bboxes_3d, 
                person_centers2d,
                vehicle_bboxes, 
                vehicle_bboxes_3d, 
                vehicle_centers2d,
                rider_bboxes, 
                rider_bboxes_3d, 
                rider_centers2d,
                rear_bboxes,
                points=concat_points,
                regress_range=concat_regress_ranges,
                num_points_per_lvl=num_points
        )
        
        person_gt = self.process_3d_ground_truth(
                    num_points,
                    num_levels,
                    person_labels,
                    person_bbox_targets,
                    person_bbox_targets_3d,
                    person_centerness2d_targets,
                    person_centerness3d_targets
             )
        vehicle_gt = self.process_3d_ground_truth(
                num_points,
                num_levels,
                vehicle_labels,
                vehicle_bbox_targets,
                vehicle_bbox_targets_3d,
                vehicle_centerness2d_targets,
                vehicle_centerness3d_targets
             )
        rider_gt = self.process_3d_ground_truth(
                num_points,
                num_levels,
                rider_labels,
                rider_bbox_targets,
                rider_bbox_targets_3d,
                rider_centerness2d_targets,
                rider_centerness3d_targets
             )
        
        rear_gt = self.process_2d_ground_truth(
            num_points,
            num_levels,
            rear_labels,
            rear_bbox_targets,
            rear_centerness2d_targets
        )
        return person_gt, vehicle_gt, rider_gt, rear_gt


    def _get_target_single(self,
                           person_bboxes,
                           person_bboxes3d,
                           person_centers2d,
                           vehicle_bboxes,
                           vehicle_bboxes3d,
                           vehicle_centers2d,
                           rider_bboxes,
                           rider_bboxes3d,
                           rider_centers2d,
                           rear_bboxes,
                           points,
                           regress_range,
                           num_points_per_lvl):
        """person_gt(List): [bbox2d, bbox_cam3d, center2ds]
                            bbox[x, y, x, y]
                            bbox_cam3d[x, y, z, w, h, l, yaw]
                            centers2d[x, y, depth]
        """

        person_labels, person_bbox_targets, person_bbox_targets_3d, person_centerness2d_targets, person_centerness3d_targets = \
            self.process_3d_target(person_bboxes, person_bboxes3d, person_centers2d, points, regress_range, num_points_per_lvl)

        vehicle_labels, vehicle_bbox_targets, vehicle_bbox_targets_3d, vehicle_centerness2d_targets, vehicle_centerness3d_targets = \
            self.process_3d_target(vehicle_bboxes, vehicle_bboxes3d, vehicle_centers2d, points, regress_range, num_points_per_lvl)

        rider_labels, rider_bbox_targets, rider_bbox_targets_3d, rider_centerness2d_targets, rider_centerness3d_targets = \
            self.process_3d_target(rider_bboxes, rider_bboxes3d, rider_centers2d, points, regress_range, num_points_per_lvl)
            
        rear_labels, rear_bbox_targets, rear_centerness2d_targets = self.process_2d_target(rear_bboxes, points, regress_range, num_points_per_lvl)

        return person_labels, person_bbox_targets, person_bbox_targets_3d, person_centerness2d_targets, person_centerness3d_targets,\
            vehicle_labels, vehicle_bbox_targets, vehicle_bbox_targets_3d, vehicle_centerness2d_targets, vehicle_centerness3d_targets,\
            rider_labels, rider_bbox_targets, rider_bbox_targets_3d, rider_centerness2d_targets, rider_centerness3d_targets,\
            rear_labels, rear_bbox_targets, rear_centerness2d_targets
            
    def _get_centerness2d_target(self, bbox_targets):
        left_right = bbox_targets[:, [0, 2]]
        top_bottom = bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        
        return torch.sqrt(centerness_targets)

    
    def process_3d_target(self, bbox, bbox3d, centers2d, points,regress_range, num_points_per_lvl):
        
        num_points = points.size(0)
        num_gt = bbox.size(0)

        if num_gt == 0:
            return torch.zeros((num_points)), torch.zeros((num_points, 4)), torch.zeros((num_points, self.bbox_code_size)),\
                 torch.zeros((num_points)), torch.zeros((num_points))
        #gt [bbox, bbox3d, centers2d]
        depths = centers2d[:, -1]
        centers2d = centers2d[:, :2]
        #change orientation to local yaw

        #bbox3d[..., 6] = -torch.atan2(bbox3d[..., 0], bbox3d[..., 2]) + bbox3d[..., 6]
        areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        areas = areas[None].repeat(num_points, 1)
        processed_regress_ranges = regress_range.clone()
        processed_regress_ranges = processed_regress_ranges[:, None, :].expand(num_points, num_gt, 2)
        
        bbox = bbox[None].expand(num_points, num_gt, 4)
        bbox3d = bbox3d[None].expand(num_points, num_gt, self.bbox_code_size)
        centers2d = centers2d[None].expand(num_points, num_gt, 2)
        depths = depths[None, :, None].expand(num_points, num_gt, 1)
        
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gt)
        ys = ys[:, None].expand(num_points, num_gt)

        delta_x = (xs - centers2d[..., 0])[..., None]
        delta_y = (ys - centers2d[..., 1])[..., None]
        bbox_targets_3d = torch.cat((delta_x, delta_y, depths, bbox3d[..., 3:]), dim=-1)

        left = xs - bbox[..., 0]
        right = bbox[..., 2] - xs
        top = ys - bbox[..., 1]
        bottom = bbox[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, 'Setting center_sampling to '\
            'False has not been implemented for FCOS3D'
        # condition1: inside a `center bbox`
        
        radius = self.center_sample_radius
        center_xs = centers2d[..., 0]    #centers2d here is the projection of 3d center point to the image actually a 2.5d point
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(bbox)
        stride = center_xs.new_zeros(center_xs.shape)

        #project the points on current lvl back to the original sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end
        
        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        #condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= processed_regress_ranges[..., 0]) &
            (max_regress_distance <= processed_regress_ranges[..., 1]))
        
        #center-based criterion to deal with ambiguity
        dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1))
        dists[inside_gt_bbox_mask == 0] = INF
        dists[inside_regress_range == 0] = INF
        min_dist, min_dist_inds = dists.min(-1)
        gt_labels = torch.tensor([1])
        labels = gt_labels[min_dist_inds]
        labels[min_dist == INF] = 0  # set as BG
        
        bbox_targets = bbox_targets[range(num_points), min_dist_inds]
        bbox_targets_3d = bbox_targets_3d[range(num_points), min_dist_inds]
        relative_dist = torch.sqrt(
            torch.sum(bbox_targets_3d[:, :2]**2, dim=-1)) / (1.414 * stride[:, 0])

        centerness3d_targets = torch.exp(-self.centerness_alpha * relative_dist)
        centerness2d_targets = self._get_centerness2d_target(bbox_targets)
        
        return labels, bbox_targets, bbox_targets_3d, centerness2d_targets, centerness3d_targets
        
            
    def process_2d_target(self, bboxes, points, regress_range, num_points_per_lvl):
        num_points = points.size(0)
        num_gts = bboxes.size(0)
        #process rear ground thuth
        if num_gts == 0:
            return torch.zeros((num_points)), torch.zeros((num_points, 4)), torch.zeros((num_points))
        else:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            areas = areas[None].repeat(num_points, 1)
            regression_ranges = regress_range.clone()
            regression_ranges = regression_ranges[:, None, :].expand(num_points, num_gts, 2)
            gt_bboxes = bboxes[None].expand(num_points, num_gts, 4)
            xs, ys = points[:, 0], points[:, 1]
            xs = xs[:, None].expand(num_points, num_gts)
            ys = ys[:, None].expand(num_points, num_gts)

            left = xs - gt_bboxes[..., 0]
            right = gt_bboxes[..., 2] - xs
            top = ys - gt_bboxes[..., 1]
            bottom = gt_bboxes[..., 3] - ys
            bbox_targets = torch.stack((left, top, right, bottom), -1)

            #center sampling
            #condition 1: inside a center bbox
            radius = self.center_sample_radius
            center_xs_2d = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys_2d = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs_2d.new_zeros(center_xs_2d.shape)

            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            
            x_mins = center_xs_2d - stride
            y_mins = center_ys_2d - stride
            x_maxs = center_xs_2d + stride
            y_maxs = center_ys_2d + stride

            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], x_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

            max_regress_distance = bbox_targets.max(-1)[0]
            inside_regress_range = (
            (max_regress_distance >= regression_ranges[..., 0])
            & (max_regress_distance <= regression_ranges[..., 1]))

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            areas[inside_gt_bbox_mask == 0] = INF
            areas[inside_regress_range == 0] = INF
            min_area, min_area_inds = areas.min(dim=1)

            gt_labels = torch.tensor([1])
            labels = gt_labels[min_area_inds]
            labels[min_area == INF] = 0  # set as BG

            bbox_targets = bbox_targets[range(num_points), min_area_inds]
            centerness2d_targets = self._get_centerness2d_target(bbox_targets)
            
            return labels, bbox_targets, centerness2d_targets


    def process_3d_ground_truth(self,
                                num_points,
                                num_levels,
                                labels_3d_list, 
                                bbox_targets_list, 
                                bbox_3d_targets_list,
                                centerness_targets_list, 
                                centerness3d_targets_list):

        #split to per img, per level
        labels_3d_list = [
            labels_3d.split(num_points, 0) for labels_3d in labels_3d_list
        ]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list
        ]
        bbox_3d_targets_list = [
            bbox_3d_targets.split(num_points, 0) for bbox_3d_targets in bbox_3d_targets_list
        ]
        centerness_targets_list = [
            centerness_targets.split(num_points, 0) for centerness_targets in centerness_targets_list
        ]
        centerness3d_targets_list = [
            centerness3d_targets.split(num_points, 0) for centerness3d_targets in centerness3d_targets_list
        ]

        #concat per level image
        concat_lvl_labels_3d = []
        concat_lvl_bbox_targets = []
        concat_lvl_bbox_3d_targets = []
        concat_lvl_centerness_targets = []
        concat_lvl_centerness3d_targets = []

        for i in range(num_levels):
            concat_lvl_labels_3d.append(
                torch.cat([labels[i] for labels in labels_3d_list])
            )
            concat_lvl_centerness_targets.append(
                torch.cat([centerness_targets[i] for centerness_targets in centerness_targets_list])
            )
            concat_lvl_centerness3d_targets.append(
                torch.cat([centerness3d_targets[i] for centerness3d_targets in centerness3d_targets_list])
            )
            bbox_targets = torch.cat([
                bbox_targets[i] for bbox_targets in bbox_targets_list
            ])
            bbox_targets_3d = torch.cat([
                bbox_targets_3d[i] for bbox_targets_3d in bbox_3d_targets_list
            ])

            if self.norm_on_box:
                bbox_targets = bbox_targets / self.strides[i]
                bbox_targets_3d[:, :2] = bbox_targets_3d[:, :2] / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_bbox_3d_targets.append(bbox_targets_3d)

        return [concat_lvl_labels_3d, concat_lvl_bbox_targets, concat_lvl_bbox_3d_targets, concat_lvl_centerness_targets, concat_lvl_centerness3d_targets]

    def process_2d_ground_truth(self,
                        num_points,
                        num_levels,
                        labels_list, 
                        bbox_targets_list, 
                        centerness_targets_list):
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        centerness_targets_list = [
                centerness_targets.split(num_points, 0) for centerness_targets in centerness_targets_list
            ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_centerness_targets = []

        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            concat_lvl_centerness_targets.append(
                torch.cat([centerness[i] for centerness in centerness_targets_list]))
            if self.norm_on_box:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        return [concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_centerness_targets]

    def forward_train(self,
             x,
             person_gt,
             vehicle_gt,
             rider_gt,
             rear_gt,
             proposal_cfg=None):
        outs = self(x)
        person_preds, vehicle_preds, rider_preds, rear_preds = outs

        loss = self.loss(
             person_preds,
             vehicle_preds,
             rider_preds,
             rear_preds,
             person_gt,
             vehicle_gt,
             rider_gt,
             rear_gt,
        )

        assert proposal_cfg is None, "not support two stage now"
        return loss
        

    def get_bboxes3d(self,
                   cls_pred,
                   bbox,
                   offset,
                   depth,
                   dim,
                   dir_cls,
                   dir_offset,
                   centerness,
                   centerness3d,
                   img_metas,
                   cfg=None):
        assert len(cls_pred) == len(bbox) == len(offset) == len(depth) == len(dim) == len(dir_cls) == \
            len(dir_offset) == len(centerness) == len(centerness3d)

        num_levels = len(cls_pred)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_pred]
        mlvl_points = self.get_points(featmap_sizes, cls_pred[0].dtype, cls_pred[0].device)


        result_list = []
        for img_id in range(len(img_metas)):
            cls_pred_list = [
                cls_pred[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox[i][img_id].detach() for i in range(num_levels)
            ]
            offset_pred_list = [
                offset[i][img_id].detach() for i in range(num_levels)
            ]
            depth_pred_list = [
                depth[i][img_id].detach() for i in range(num_levels)
            ]
            dim_pred_list = [
                dim[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls[i][img_id].detach() for i in range(num_levels)
            ]
            dir_offset_pred_list = [
                dir_offset[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centerness[i][img_id].detach() for i in range(num_levels)
            ]
            centerness3d_pred_list = [
                centerness3d[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = img_metas[img_id]
            det_bboxes = self._get_bboxes3d_single(
                cls_pred_list, bbox_pred_list, offset_pred_list, depth_pred_list, dim_pred_list,
                dir_cls_pred_list, dir_offset_pred_list, centerness_pred_list, centerness3d_pred_list,
                mlvl_points, input_meta, cfg
            )
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes3d_single(self,
                            cls_scores,
                            bboxes,
                            offsets,
                            depths,
                            dims,
                            dir_scores,
                            dir_offsets,
                            centernesses,
                            centernesses3d,
                            mlvl_points,
                            input_meta,
                            cfg):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bboxes) == len(offsets) == len(depths) == len(dims) == \
            len(dir_scores) == len(dir_offsets) == len(centernesses) == len(centernesses3d) == len(mlvl_points)

        mlvl_centers2d = []
        mlvl_bboxes = []
        mlvl_cls_scores = []
        mlvl_offsets = []
        mlvl_depths = []
        mlvl_dims = []
        mlvl_dir_scores = []
        mlvl_dir_offsets = []
        mlvl_centernesses = []
        mlvl_centernesses3d = []

        for cls_score, bbox, offset, depth, dim, dir_score, dir_offset, centerness, centerness3d, points in \
            zip(cls_scores, bboxes, offsets, depths, dims, dir_scores, dir_offsets, centernesses, centernesses3d, mlvl_points):
            assert cls_score.size()[-2:] == bbox.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox = bbox.permute(1, 2, 0).reshape(-1, 4)
            offset = offset.permute(1, 2, 0).reshape(-1, 2)
            depth = depth.permute(1, 2, 0).reshape(-1)
            dim = dim.permute(1, 2, 0).reshape(-1, 3)
            dir_score = dir_score.permute(1, 2, 0).reshape(-1, 4)
            dir_offset = dir_offset.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permuter(1, 2, 0).reshape(-1).sigmoid()
            centerness3d = centerness3d.permuter(1, 2, 0).reshape(-1).sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:  #TODO check scores.shape[0]
                max_scores, _ = (scores * centerness3d[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds]
                depth = depth[topk_inds]
                dim = dim[topk_inds, :]


        





