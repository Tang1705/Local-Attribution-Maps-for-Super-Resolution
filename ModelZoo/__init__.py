from .sr_backbones import sisr
from .sr_backbones import vsr


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
    assert model_name in sisr.MODEL_LIST or model_name in vsr.MODEL_LIST, 'check your model name before @'
    if model_name in sisr.MODEL_LIST:
        return sisr.load_model(model_name,training_name)
    else:
        return vsr.load_model(model_name,training_name)