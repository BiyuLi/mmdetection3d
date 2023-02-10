model = dict(
    type='HppFCOSMono3D',
    backbone=dict(
        type='CSPDarknet',
        arch='P6',
        widen_factor=0.25,
        out_indices=range(2, 6),
        spp_kernal_sizes=(5, 9, 13)),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 192, 256],
        out_channels=64,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='HppFCOSOneStageMono3DHead',
        num_classes=4,
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
        num_direction_classfier=4,
        diff_rad_by_sin=True,
        strides=(8, 16, 32, 64, 128),
        group_reg_dims=(4, 1),
        group_reg_dims_3d=(4, 2, 1, 3, 4, 4, 1, 1),
        cls_branch=(64,),
        reg_branch=((64,),
                (64,)),
        dir_branch=(64,),
        centerness_branch=(64,),
        centerness3d_branch=(64,),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_iou=dict(
            type='CIoULoss',
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=1.0/9.0,
            loss_weight=1.0
        ),
        loss_dir=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        ),
        loss_centerness=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
        loss_centerness3d=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
        loss_rot=dict(
            type='SmoothL1Loss',
            beta=1.0/9.0,
            loss_weight=1.0
        ),
        bbox_coder=dict(type='FCOS3DBBoxCoder', code_size=7),
        bbox_code_size=7,
        norm_on_bbox=True,
        centerness_on_reg=True,
        centerness_alpha=2.5,
        center_sampling=True,
        center_sample_radius=1.5,
        dir_offset=0.7854,
        dir_limit_offset=0,
        conv_bias=True,
        background_label=0     
        ),
    train_cfg=dict(
        allowed_border=0,
        code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=200))
