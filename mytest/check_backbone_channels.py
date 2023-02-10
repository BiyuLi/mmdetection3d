import torch
import numpy
import mmcv
from mmdet3d.models import build_backbone


if __name__ == "__main__":
    # backbone=dict(
    #     type='EfficientNet',
    #     arch='b0',
    #     drop_path_rate=0.2,
    #     out_indices=(0, 1, 2, 3, 4, 5, 6),
    #     frozen_stages=0,
    #     norm_cfg=dict(
    #         type='BN', requires_grad=True, eps=1e-3, momentum=0.01),
    #     norm_eval=False,
    #     init_cfg=dict(
    #         type='Pretrained', prefix='backbone', checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth')
    #     )
    
    backbone=dict(
        type='MobileNetV2',
        out_indices=(0, 1, 2, 3,4, 5,6,7),
        norm_cfg=dict(type='BN', requires_grad=False),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2'))
    img = torch.randn((1, 3, 320,320))
    backbone = build_backbone(backbone)
    output = backbone(img)
    for i in output:
        print(i.shape)
    print('process done!!!')