_base_ = './co_dino_5scale_r50_lsj_8xb2_1x_lvis-v1.py'
# --- Visualizer: Local + W&B backends (LVIS v1) ---
visualizer = dict(
    _delete_=True,  # remove this line if your _base_ has no `visualizer`
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='codetr_lvis-v1',    # <— LVIS-friendly project name
                entity='sidgairo18-saarland-informatics-campus'
            ),
            # Use LVIS metric names instead of COCO
            define_metric_cfg=[
                # bbox AP (primary LVIS metric)
                dict(name='lvis/bbox_mAP', summary='max', step_metric='epoch'),
                # segm AP (only if you added a mask head; otherwise harmless to leave)
                dict(name='lvis/segm_mAP', summary='max', step_metric='epoch'),
                # training curves
                dict(name='train/loss', summary='min', step_metric='iter'),
                dict(name='lr',           step_metric='iter'),
            ],
            # Optional niceties:
            # log_code_name='source',
            # watch_kwargs=dict(log='gradients')
        )
    ],
    name='visualizer'
)

# --- Hooks ---
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    # If you want to draw images to W&B/Local, flip draw=True and set a reasonable interval
    visualization=dict(type='DetVisualizationHook', draw=False),
    # Save best checkpoint by LVIS bbox mAP (change to 'lvis/segm_mAP' if you care about masks)
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=3, save_best='lvis/bbox_mAP')
)

# If you register custom hooks elsewhere, keep as-is:
'''
custom_hooks = [
    # dict(type='WandbAutoMetaHook', priority='ABOVE_NORMAL'),
    # dict(type='WandbArtifactHook', priority='VERY_LOW'),
]
'''

log_processor = dict(by_epoch=True)
