from .DDRNet_23_slim import get_ddrnet_23_slim
from .DDRNet_39 import get_ddrnet_39
from .DDRNet_23 import get_ddrnet_23


models = {
    'ddrnet_39': get_ddrnet_39,
    'ddrnet_23_slim': get_ddrnet_23_slim,
    'ddrnet_23': get_ddrnet_23
}


def get_segmentation_model(model, **kwargs):
    """Segmentation models"""
    return models[model.lower()](**kwargs)
