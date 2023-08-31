import torch
import torch.nn as nn

class MslmModel(nn.Module):
    def __init__(self, model):
        super(MslmModel, self).__init__()
        self.model = model

    def forward(self, batch_input):
        batch_input = {k: v for k, v in batch_input.items() if k not in ['entity_specific_mask_ids',
                                                                         'non_entity_specific_mask_ids',
                                                                         'input_ids']}
        batch_input['input_ids'] = batch_input.pop('masked_input_ids')
        output = self.model(**batch_input, output_hidden_states=True)
        return output

