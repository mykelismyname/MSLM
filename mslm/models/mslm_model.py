import torch
import torch.nn as nn

class MslmModel(nn.Module):
    def __init__(self, model):
        super(MslmModel, self).__init__()
        self.model = model

    def forward(self, batch_input):
        output = self.model(**batch_input, output_hidden_states=True)
        return output

