_base_ = './mask-rcnn_r50_fpn_2x_coco.py'

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(type='AmpOptimWrapper')

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))
