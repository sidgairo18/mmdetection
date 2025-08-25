_base_ = './co_dino_5scale_r50_lsj_8xb2_1x_lvis-v1.py'

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        dataset=dict(
            ann_file='annotations/lvis_v1_minitrain_0.25.json')))
