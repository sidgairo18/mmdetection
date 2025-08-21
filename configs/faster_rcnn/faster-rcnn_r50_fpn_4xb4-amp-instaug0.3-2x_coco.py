_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

backend_args = None
inst_augment_base_dir = '/BS/generative_modelling_for_image_understanding/nobackup/data/DETECTRON2_DATASETS/coco_augmented/'
inst_augment_prob = 0.3
train_pipeline = [
    dict(type='LoadImageFromFileInstAug',
         inst_augment_base_dir = inst_augment_base_dir,
         inst_augment_p = inst_augment_prob,
         backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        pipeline=train_pipeline,
        backend_args=backend_args))

# Enable automatic-mixed-precision training with AmpOptimWrapper.                             
optim_wrapper = dict(type='AmpOptimWrapper')                                                  
                                                                                              
default_hooks = dict(checkpoint=dict(max_keep_ckpts=2)) 
