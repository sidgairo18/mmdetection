_base_ = './co_dino_5scale_r50_4xb4_1x_lvis-v1.py'


train_dataloader = dict(
    batch_size=3,
    num_workers=2,
    dataset=dict(
        ann_file='annotations/lvis_v1_minitrain_0.25.json'))
