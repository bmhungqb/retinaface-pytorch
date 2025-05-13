def get_config(network):
    configs = {
        "mobilenetv1": cfg_mnet,
        "mobilenetv1_0.25": cfg_mnet_025,
        "mobilenetv1_0.50": cfg_mnet_050,
        "mobilenetv2": cfg_mnet_v2,
        "mobilenetv2_0.125": cfg_mnet_v2_0125,
        "mobilenetv2_0.25": cfg_mnet_v2_025,
        "mobilenetv3": cfg_mnet_v3,
        "mobilenetv3_0.125": cfg_mnet_v3_0125,
        "mobilenetv3_0.25": cfg_mnet_v3_025,
        # "mobilenetv4": cfg_mnet_v4,
        # "mobilenetv4_0.125": cfg_mnet_v4_0125,
        # "mobilenetv4_0.25": cfg_mnet_v4_025,
        "resnet50": cfg_re50,
        "resnet34": cfg_re34,
        "resnet18": cfg_re18
    }
    return configs.get(network, None)


cfg_mnet_025 = {
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
    'in_channel': 32,
    'out_channel': 64
}

cfg_mnet_050 = {
    'name': 'mobilenet0.50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 64,
    'out_channel': 128
}

cfg_mnet = {
    'name': 'mobilenet_v1',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 128,
    'out_channel': 128
}

cfg_mnet_v2 = {
    'name': 'mobilenet_v2',
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
    'return_layers': [6, 13, 18],
    'in_channel': 32,
    'out_channel': 128
}
cfg_mnet_v2_0125 = {
    'name': 'mobilenet_v2_0.125',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 320,
    'pretrain': True,
    'return_layers': [6, 13, 18],
    'in_channel': 8,
    'out_channel': 64
}
cfg_mnet_v2_025 = {
    'name': 'mobilenet_v2_0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
    'pretrain': False,
    'return_layers': [6, 13, 18],
    'in_channel': 8,
    'out_channel': 64
}

# Mobilenetv3
cfg_mnet_v3 = {
    'name': 'mobilenet_v3',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'bneck1': 1, 'bneck2': 2, 'bneck3': 3},
    'in_channel': 32,
    'out_channel': 128
}
cfg_mnet_v3_025 = {
    'name': 'mobilenetv3_0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 320,
    'pretrain': True,
    'return_layers': {'bneck1': 1, 'bneck2': 2, 'bneck3': 3},
    'in_channel': 8,
    'out_channel': 64
}
cfg_mnet_v3_0125 = {
    'name': 'mobilenetv3_0.125',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': True,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 320,
    'pretrain': True,
    'return_layers': {'bneck1': 1, 'bneck2': 2, 'bneck3': 3},
    'in_channel': 4,
    'out_channel': 64
}

cfg_re18 = {
    'name': 'resnet18',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 150,
    'milestones': [70, 90],
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 64,
    'out_channel': 128
}
cfg_re34 = {
    'name': 'resnet34',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 100,
    'milestones': [70, 90],
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 64,
    'out_channel': 128
}

cfg_re50 = {
    'name': 'resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 8,
    'epochs': 100,
    'milestones': [70, 90],
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}
