import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

class Detection_model(BasicModule):
    def __init__(self, hidden_dim, token_types, drop_out):
        super(Detection_model, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_out)
        self.entity_classifier = nn.Linear(hidden_dim, len(token_types))

    def prepare_embeddings(self, all_layer_hidd, mode='last'):
        # last layer
        if mode == 'last':
            emb = all_layer_hidd[-1]
            return emb
        #average embeddings across all model layers
        elif mode == 'average':
            all_layer_hidd = [layer_hidd for layer_hidd in all_layer_hidd[1:]]
            all_layer_hidd = torch.stack(all_layer_hidd, dim=0)
            all_layer_hidd = torch.mean(all_layer_hidd, dim=0)
            return all_layer_hidd

    def forward(self, input):
        input_hidden_states = input.get("hidden_states")
        token_hidden_states = self.prepare_embeddings(input_hidden_states)
        output = self.dropout(self.linear(token_hidden_states))
        det_classifier_output = self.entity_classifier(output)
        det_output = torch.softmax(det_classifier_output, dim=2)
        return det_output