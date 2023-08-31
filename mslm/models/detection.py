import torch
import torch.nn as nn

class DetectionModel(nn.Module):
    def __init__(self, config, token_types, drop_out, meta_embedding_dim):
        super(DetectionModel, self).__init__()
        self.hidden_dim = config.hidden_size
        self.meta_embedding_dim = meta_embedding_dim
        self.meta_embeddings = nn.Embedding(config.vocab_size, meta_embedding_dim)
        self.linear = nn.Linear(self.hidden_dim+meta_embedding_dim, self.hidden_dim)
        self.dropout = nn.Dropout(drop_out)
        self.entity_classifier = nn.Linear(self.hidden_dim, len(token_types))

    def prepare_embeddings(self, all_layer_hidd, mode='last'):
        #last layer
        if mode == 'last':
            emb = all_layer_hidd[-1]
            return emb
        #average embeddings across all model layers
        elif mode == 'average':
            all_layer_hidd = [layer_hidd for layer_hidd in all_layer_hidd[1:]]
            all_layer_hidd = torch.stack(all_layer_hidd, dim=0)
            all_layer_hidd = torch.mean(all_layer_hidd, dim=0)
            return all_layer_hidd

    def forward(self, input, hidden_states):
        token_hidden_states = self.prepare_embeddings(hidden_states)
        input_ids = input.get('input_ids')
        b_size, seq_len = input_ids.size()
        input_ids = input_ids.view(-1)
        meta_embeddings = self.meta_embeddings(input_ids)
        meta_embeddings = meta_embeddings.view(b_size, seq_len, self.meta_embedding_dim)
        meta_hidden_states = torch.cat([token_hidden_states, meta_embeddings], dim=2)
        meta_hidden_states = self.dropout(torch.relu(self.linear(meta_hidden_states)))
        det_classifier_output = self.entity_classifier(meta_hidden_states)
        # det_classifier_output = torch.softmax(det_classifier_output, dim=2)
        return det_classifier_output