import torch
import torch.nn as nn
import torch.nn.functional as F
from mslm.models.detection import DetectionModel
from mslm.models.mslm_model import MslmModel
from mslm.loss import MslmLoss, DetectionLoss
from typing import Optional, Tuple

class Output:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

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
    def __init__(self, model, config, token_types, drop_out, reduction, batch_size, max_length, meta_embedding_dim):
        super().__init__()
        self.config = config
        self.model = model
        self.mslm_model = MslmModel(self.model)
        self.detection = DetectionModel(config=config,
                                        token_types=token_types,
                                        drop_out=drop_out,
                                        meta_embedding_dim=meta_embedding_dim)
        self.mslm_loss = MslmLoss(batch_size, max_length, config.vocab_size)
        self.det_loss = DetectionLoss(len(token_types))
        self.reduction = reduction
        self.output = Output()

    def forward(self, input, labels, ms_weights, weight_matrix, train=True):
        mslm_output = self.mslm_model(input)
        mslm_labels = input['input_ids']
        input_size = mslm_labels.size(0)
        ms_weights = ms_weights[:input_size]
        mslm_hidden_states = mslm_output.get("hidden_states")
        det_output = self.detection(input, mslm_hidden_states)
        if train:
            mslm_loss = self.mslm_loss.compute_loss(mslm_output, mslm_labels, ms_weights, weight_matrix,
                                                    reduction=self.reduction)
            det_loss = self.det_loss.compute_loss(det_output, labels, ms_weights, weight_matrix,
                                                  reduction=self.reduction)
            loss = mslm_loss + det_loss
            self.output.loss = loss
        self.output.logits = det_output
        self.output.hidden_states = mslm_hidden_states
        return self.output