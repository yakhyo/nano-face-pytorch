def get_config(network):
    configs = {
        "mobilenetv1_0.25": cfg_mnet,
        "slim": cfg_slim,
        "rfb": cfg_rfb,
    }
    return configs.get(network, None)


cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 64,
    'out_channel': 64
}

cfg_slim = {
    'name': 'slim',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
}

cfg_rfb = {
    'name': 'rfb',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
}
