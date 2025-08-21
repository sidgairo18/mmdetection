_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

train_dataloader = dict(                                                                      
    batch_size=4,                                                                             
    num_workers=2)                                                                            
                                                                                              
# Enable automatic-mixed-precision training with AmpOptimWrapper.                             
optim_wrapper = dict(type='AmpOptimWrapper')                                                  
                                                                                              
default_hooks = dict(checkpoint=dict(max_keep_ckpts=2)) 
