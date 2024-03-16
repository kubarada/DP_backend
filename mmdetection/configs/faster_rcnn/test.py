_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
# fp16 settings
fp16 = dict(loss_scale=512.)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('pig_shape',)
data = dict(
    train=dict(
        img_prefix='D:/coco_annotations_yolo/images/',
        classes=classes,
        ann_file='D:/coco_annotations_yolo/annotations/instances_default.json'),
    val=dict(
        img_prefix='D:/bp_annotations_folder_yolo/valid/frames/',
        classes=classes,
        ann_file='D:/bp_annotations_folder_yolo/valid/json/instances_default.json'),
    test=dict(
        img_prefix='D:/bp_annotations_folder_yolo/test/frames/',
        classes=classes,
        ann_file='D:/bp_annotations_folder_yolo/test/json/instances_default.json'))

