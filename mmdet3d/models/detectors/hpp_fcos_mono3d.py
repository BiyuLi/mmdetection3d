# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage_mono3d import SingleStageMono3DDetector


@DETECTORS.register_module()
class HppFCOSMono3D(SingleStageMono3DDetector):
    r"""`FCOS3D <https://arxiv.org/abs/2104.10956>`_ for monocular 3D object detection.

    Currently please refer to our entry on the
    `leaderboard <https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera>`_.
    """  # noqa: E501

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(HppFCOSMono3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)

    def forward_train(self,
                      img,
                      person_gt,
                      vehicle_gt,
                      rider_gt,
                      rear_gt):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): Each item are the 3D truth boxes for
                each image in [x, y, z, x_size, y_size, z_size, yaw, vx, vy]
                format.
            gt_labels_3d (list[Tensor]): 3D class indices corresponding to
                each box.
            centers2d (list[Tensor]): Projected 3D centers onto 2D images.
            depths (list[Tensor]): Depth of projected centers on 2D images.
            attr_labels (list[Tensor], optional): Attribute indices
                corresponding to each box
            gt_bboxes_ignore (list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, person_gt, vehicle_gt, 
                                              rider_gt, rear_gt)
        return losses