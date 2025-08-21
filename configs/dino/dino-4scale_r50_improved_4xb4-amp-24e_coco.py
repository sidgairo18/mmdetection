_base_ = ['dino-4scale_r50_8xb2-12e_coco.py']

# from deformable detr hyper
model = dict(
    backbone=dict(frozen_stages=-1),
    bbox_head=dict(loss_cls=dict(loss_weight=2.0)),
    positional_encoding=dict(offset=-0.5, temperature=10000),
    dn_cfg=dict(group_cfg=dict(num_dn_queries=300)))

train_dataloader = dict(
    batch_size=4,
    num_workers=2)

# Enable automatic-mixed-precision training with AmpOptimWrapper.
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(lr=0.0002),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

max_epochs = 24                                                                               
train_cfg = dict(                                                                             
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)                        
param_scheduler = [                                                                           
    dict(                                                                                     
        type='MultiStepLR',                                                                   
        begin=0,                                                                              
        end=max_epochs,                                                                       
        by_epoch=True,                                                                        
        milestones=[20],                                                                      
        gamma=0.1)                                                                            
]

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))
auto_scale_lr = dict(enable=False, base_batch_size=16)
