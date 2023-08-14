from abc import ABC
import torch

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

class MslmLoss(Loss):
    def __init__(self, dataset, batch_size, seq_len, *args, **kwargs):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self._softmax = torch.nn.Softmax(dim=0)
        tracked_dataset_ids = self.identify_tokens_with_and_without_masks(dataset)
        self.entity_mask = tracked_dataset_ids[0]
        self.non_entity_mask = tracked_dataset_ids[1]
        self.non_masked = tracked_dataset_ids[2]
        self.computed_weights = self.compute_weights(self.entity_mask,
                                                     self.non_entity_mask,
                                                     self.non_masked)
        self.weights = self.compute_mask_specific_weights()


    #retrieve tensors with ids of entity-level, base-level masks as well as non-masked tokens
    def identify_tokens_with_and_without_masks(self, dataset):
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
        return entity_mask, non_entity_mask, non_masked

    #calculate a weight for the arbitrary masked tokens (base level masking), entity masked tokens
    # (entity-level masking) as well as non-masked tokens
    def compute_weights(self, e_ms, ne_ms, n_ms):
        def flatten_batch_of_lists(x):
            output = []
            for batch in x:
                for seq in batch:
                    for index in seq:
                        output.append(index)
            return output

        len_e_ms = len(flatten_batch_of_lists(e_ms))
        len_ne_ms = len(flatten_batch_of_lists(ne_ms))
        len_n_ms = len(flatten_batch_of_lists(n_ms))
        total_len = len_e_ms + len_ne_ms + len_n_ms

        self.weight_matrix = torch.zeros(3)
        for n,m in enumerate([len_e_ms, len_ne_ms, len_n_ms]):
            self.weight_matrix[n] = 1 - float(m/total_len)
        return self._softmax(self.weight_matrix)

    #compute a tensor with a weight for each token with respect to the mask (mask specific weight)
    def compute_mask_specific_weights(self):
        assert len(self.entity_mask) == len(self.non_entity_mask) == len(self.non_masked), "Something is wrong"
        n = len(self.entity_mask)
        weights = torch.zeros(n, self.batch_size, self.seq_len)
        for b in range(n):
            for i in range(len(self.entity_mask[b])):
                weights[b][i, self.entity_mask[b][i]] = self.computed_weights[0]
                weights[b][i, self.non_entity_mask[b][i]] = self.computed_weights[1]
                weights[b][i, self.non_masked[b][i]] = self.computed_weights[2]
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