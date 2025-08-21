_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 5
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))


train_dataloader = dict(                                                                      
    batch_size=4,                                                                             
    num_workers=2)                                                                            
                                                                                              
# Enable automatic-mixed-precision training with AmpOptimWrapper.                             
# optimizer                                                                                   
optim_wrapper = dict(                                                                         
    type='AmpOptimWrapper',                                                                      
    optimizer=dict(                                                                           
        type='AdamW',                                                                         
        lr=0.0001,  # 0.0002 for DeformDETR                                                   
        weight_decay=0.0001),                                                                 
    clip_grad=dict(max_norm=0.1, norm_type=2),                                                
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})                           
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa         
     

                                                                                              
max_epochs = 12
train_cfg = dict(                                                                             
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)                        
param_scheduler = [                                                                           
    dict(                                                                                     
        type='MultiStepLR',                                                                   
        begin=0,                                                                              
        end=max_epochs,                                                                       
        by_epoch=True,                                                                        
        milestones=[11],                                                                      
        gamma=0.1)                                                                            
]                                                                                             
                                                                                              
default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))                                       
auto_scale_lr = dict(enable=False, base_batch_size=16)
