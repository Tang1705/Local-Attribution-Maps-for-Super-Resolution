import os
import torch
from ...utils import print_network

WEIGHT_DIR = './ModelZoo/weights/'

MODEL_LIST = [
    'EDVR',
    'BASICVSR',
    'ICONVSR',
    'BASICVSRPP',
    'VRT',
    'TTVSR',
    'RVRT',
    'PSRT',
    'SemanticLens'
]

MODEL_DICT = {
    'EDVR': {
        'Base': 'edvr_reds4.pth',
    },
    'BASICVSR': {
        'Base': 'basicvsr_reds4.pth',
    },
    'ICONVSR': {
        'Base': 'iconvsr_reds4.pth',
    },
    'BASICVSRPP': {
        'Base': 'basicvsr_plus_plus_reds4.pth',
    },
    'TTVSR': {
        'Base': 'ttvsr_reds4.pth',
    },
    'VRT': {
        'Base': 'vrt_reds4.pth',
    },
    'RVRT': {
        'Base': 'rvrt_reds4.pth',
    },
    'PSRT': {
        'Base': 'psrt_reds4.pth',
    },
    'SemanticLens': {
        'Base': 'Base VSR BI.pth',
        'Full': 'Semantic Lens.pth'
    }
}


def get_model(model_name, factor=4, num_channels=3):
    """
    All the weights are defaulted to be X4 weights, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in MODEL_LIST:

        if model_name == 'EDVR':
            from .edvr_net import EDVRNet
            net = EDVRNet(in_channels=num_channels, out_channels=num_channels)

        elif model_name == 'BASICVSR':
            from .basicvsr_net import BasicVSRNet
            net = BasicVSRNet()

        elif model_name == 'ICONVSR':
            from .iconvsr_net import IconVSRNet
            net = IconVSRNet()

        elif model_name == 'BASICVSRPP':
            from .basicvsr_plus_plus import BasicVSRPlusPlusNet
            net = BasicVSRPlusPlusNet()

        elif model_name == 'TTVSR':
            from .ttvsrnet import TTVSRNet
            net = TTVSRNet()

        elif model_name == 'VRT':
            from .vrt import VRT
            net = VRT(upscale=4, img_size=[6, 64, 64], window_size=[6, 8, 8],
                      depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
                      indep_reconsts=[11, 12],
                      embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
                      num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=12)

        elif model_name == 'RVRT':
            from .rvrt import RVRT
            net = RVRT(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                       depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                       inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                       attention_window=[3, 3], cpu_cache_length=100)

        elif model_name == 'PSRT':
            from .psrt_recurrent_arch import BasicRecurrentSwin
            net = BasicRecurrentSwin()

        elif model_name == "SemanticLens":
            from ModelZoo.sr_backbones.semantic_lens.super_resolution import VISR
            net = VISR()

        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_name, training_name="Base"):
    """
    :param model_name: model_name
    :param training_name: training_name
    :return:
    """
    net = get_model(model_name)
    state_dict_path = os.path.join(WEIGHT_DIR, MODEL_DICT[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    net.load_state_dict(state_dict)
    return net