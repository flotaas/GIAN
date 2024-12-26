"""
AIO -- All Model in One
"""
import torch.nn as nn
from .HGCN import HGCN
from .HGCN_ablation import HGCN_wo_TGCN, HGCN_wo_MGCN, HGCN_wo_fusion
__all__ = ['AMIO']

MODEL_MAP = {
    'HGCN': HGCN,
    'HGCN_wo_TGC': HGCN_wo_TGCN,
    'HGCN_wo_MGC': HGCN_wo_MGCN,
    'HGCN_wo_fusion': HGCN_wo_fusion
}

class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x):
        return self.Model(text_x, audio_x, video_x)