import torch
import torch.nn as nn
import torch.nn.functional as F
from mslm.models.detection import DetectionModel
from mslm.models.mslm_model import MslmModel

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path

class JointModel(BasicModule):
    def __init__(self, model, config, token_types, drop_out, meta_embedding_dim):
        super().__init__()
        self.model = model
        self.mslm_model = MslmModel(self.model)
        self.detection = DetectionModel(config=config,
                                        token_types=token_types,
                                        drop_out=drop_out,
                                        meta_embedding_dim=meta_embedding_dim)

    def forward(self, input):
        mslm_output = self.mslm_model(input)
        mslm_hidden_states = mslm_output.get("hidden_states")
        det_output = self.detection(input, mslm_hidden_states)
        return mslm_output, det_output
