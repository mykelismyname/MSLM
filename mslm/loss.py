from abc import ABC
import torch

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

class MslmLoss(Loss):
    def __init__(self, batch_size, seq_len, *args, **kwargs):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self._softmax = torch.nn.Softmax(dim=0)

    #retrieve tensors with ids of entity-level, base-level masks as well as non-masked tokens
    def identify_tokens_with_and_without_masks(self, datasets):
        datasets_masked_ids = []
        for dataset in datasets:
            entity_mask = []
            non_entity_mask = []
            non_masked = []
            for d in dataset:
                batch_size = d['input_ids'].size(0)
                d_entity_mask, d_non_entity_mask, d_non_masked = [], [], []
                for k in range(batch_size):
                    entity_mask_no = d['entity_specific_mask_ids'][k]
                    non_entity_mask_no = d['non_entity_specific_mask_ids'][k]
                    entity_mask_no = entity_mask_no > 0
                    entity_mask_no_list = torch.flatten(entity_mask_no.nonzero()).tolist()
                    non_entity_mask_no = non_entity_mask_no > 0
                    non_entity_mask_no_list = torch.flatten(non_entity_mask_no.nonzero()).tolist()
                    non_mask_list = [i for i in range(len(entity_mask_no)) if
                                     i not in entity_mask_no_list + non_entity_mask_no_list]
                    d_entity_mask.append(entity_mask_no_list)
                    d_non_entity_mask.append(non_entity_mask_no_list)
                    d_non_masked.append(non_mask_list)
                entity_mask.append(d_entity_mask)
                non_entity_mask.append(d_non_entity_mask)
                non_masked.append(d_non_masked)
            datasets_masked_ids.append([entity_mask, non_entity_mask, non_masked])
        return datasets_masked_ids

    #calculate a weight for the arbitrary masked tokens (base level masking), entity masked tokens
    # (entity-level masking) as well as non-masked tokens
    def compute_weights(self, train_ids, eval_ids):
        def flatten_batch_of_lists(x):
            output = []
            for batch in x:
                for seq in batch:
                    for index in seq:
                        output.append(index)
            return output

        len_e_ms, len_ne_ms, len_n_ms = 0, 0, 0 #number of entity tokens masked, arbitrary masked tokens and non masked tokens
        for d in [train_ids, eval_ids]:
            e, f, g = d
            len_e_ms += len(flatten_batch_of_lists(e))
            len_ne_ms += len(flatten_batch_of_lists(f))
            len_n_ms += len(flatten_batch_of_lists(g))

        print(f"Number of entity tokens masked: {len_e_ms}")
        print(f"Number of arbtrary masked tokens: {len_ne_ms}")
        print(f"Number of non masked tokens: {len_n_ms}")
        total_len = len_e_ms + len_ne_ms + len_n_ms

        self.weight_matrix = torch.zeros(3)
        for n,m in enumerate([len_e_ms, len_ne_ms, len_n_ms]):
            self.weight_matrix[n] = 1 - float(m/total_len)
        return self._softmax(self.weight_matrix)

    #compute a tensor with a weight for each token with respect to the mask (mask specific weight)
    def compute_mask_specific_weights(self, data, weight_matrix):
        entity_masked_data, non_entity_masked_data, non_masked_data = data
        assert len(entity_masked_data) == len(non_entity_masked_data) == len(non_masked_data), "Something is wrong"
        n = len(entity_masked_data)
        weights = torch.zeros(n, self.batch_size, self.seq_len)
        for b in range(n):
            for i in range(len(entity_masked_data[b])):
                weights[b][i, entity_masked_data[b][i]] = weight_matrix[0]
                weights[b][i, non_entity_masked_data[b][i]] = weight_matrix[1]
                weights[b][i, non_masked_data[b][i]] = 1
        return weights

    #compute a cross entropy loss
    def compute_loss(self, inputs, labels, mask_specific_weights, reduction='none', **kwargs):
        inputs = inputs.get("logits")
        inputs = inputs.view(-1, inputs.size(-1))
        labels = labels.view(-1)
        loss = self._criterion(inputs, labels)
        loss = torch.mul(loss, mask_specific_weights.view(-1))
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            pass
        return loss

class DetectionLoss(Loss):
    def __init__(self, *args, **kwargs):
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')

    #compute a cross entropy loss
    def compute_loss(self, inputs, labels, reduction='none', **kwargs):
        inputs = inputs.view(-1, inputs.size(-1))
        labels = labels.view(-1)
        loss = self._criterion(inputs, labels)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            pass
        return loss