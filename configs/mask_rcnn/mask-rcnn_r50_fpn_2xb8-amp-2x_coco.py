_base_ = './mask-rcnn_r50_fpn_2x_coco.py'

train_dataloader = dict(
    batch_size=8,
    num_workers=4)

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))
