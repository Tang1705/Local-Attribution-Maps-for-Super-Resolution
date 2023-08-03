import os
import torch

MODEL_DIR = './ModelZoo/models'

NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN',
    'SAN',
    'EDVR',
    'BASICVSR',
    'ICONVSR',
    'BASICVSRPP',
    'TTVSR',
    'PSRT'
]

MODEL_LIST = {
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
    'PSRT': {
        'Base': 'psrt_reds4.pth',
    },
}


def print_network(model, model_name):
    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'RCAN':
            from .NN.rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .NN.rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .NN.san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .NN.rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)

        elif model_name == 'EDVR':
            from .NN.edvr_net import EDVRNet
            net = EDVRNet(in_channels=3, out_channels=3)

        elif model_name == 'BASICVSR':
            from .NN.basicvsr_net import BasicVSRNet
            net = BasicVSRNet()

        elif model_name == 'ICONVSR':
            from .NN.iconvsr_net import IconVSRNet
            net = IconVSRNet()

        elif model_name == 'BASICVSRPP':
            from .NN.basicvsr_plus_plus import BasicVSRPlusPlusNet
            net = BasicVSRPlusPlusNet()

        elif model_name == 'TTVSR':
            from .NN.ttvsrnet import TTVSRNet
            net = TTVSRNet()

        elif model_name == 'PSRT':
            from .NN.psrt_recurrent_arch import BasicRecurrentSwin
            net = BasicRecurrentSwin()

        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    net.load_state_dict(state_dict, strict=False)
    return net
