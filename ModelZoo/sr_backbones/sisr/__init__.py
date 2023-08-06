import os
import torch
from ...utils import print_network

WEIGHT_DIR = './ModelZoo/weights/'

MODEL_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN',
    'SAN',
]

MODEL_DICT = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
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

        if model_name == 'RCAN':
            from .rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)

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