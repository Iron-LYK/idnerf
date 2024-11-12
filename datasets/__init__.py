from .dtu import MVSDatasetDTU
from .llff import MVSDatasetRealFF
from .colmap import MVSDatasetCOLMAP


datas_dict = {
    'dtu': MVSDatasetDTU,
    'llff': MVSDatasetRealFF,
    'colmap': MVSDatasetCOLMAP,
}
