_base_ = '/storage/plzen1/home/jarada/mmdetection/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))
classes = ('pig_shape',)
data = dict(
    train=dict(
        img_prefix='/storage/plzen1/home/jarada/coco_annotations_yolo/images/',
        classes=classes,
        ann_file='/storage/plzen1/home/jarada/coco_annotations_yolo/annotations/instances_default.json'),
    val=dict(
        img_prefix='/storage/plzen1/home/jarada/bp_annotations_folder_yolo/valid/frames/',
        classes=classes,
        ann_file='/storage/plzen1/home/jarada/bp_annotations_folder_yolo/valid/json/instances_default.json'),
    test=dict(
        img_prefix='/storage/plzen1/home/jarada/bp_annotations_folder_yolo/test/frames/',
        classes=classes,
        ann_file='/storage/plzen1/home/jarada/bp_annotations_folder_yolo/test/json/instances_default.json'))