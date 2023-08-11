from abc import ABC
import torch

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

class MslmLoss(Loss):
    def __init__(self, dataset, seq_len, *args, **kwargs):
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self._softmax = torch.nn.Softmax(dim=0)
        tracked_dataset_ids = self.identify_tokens_with_and_without_masks(dataset)
        self.entity_mask = tracked_dataset_ids[0]
        self.non_entity_mask = tracked_dataset_ids[1]
        self.non_masked = tracked_dataset_ids[2]
        self.computed_weights = self.compute_weights(self.entity_mask,
                                                     self.non_entity_mask,
                                                     self.non_masked)
        self.weights = self.compute_mask_specific_weights(seq_len)

    #retrieve tensors with ids of entity-level, base-level masks as well as non-masked tokens
    def identify_tokens_with_and_without_masks(self, dataset):
        entity_mask = []
        non_entity_mask = []
        non_masked = []
        s = 0
        for d in dataset:
            entity_mask_no = torch.tensor(d['entity_specific_mask_ids'])
            non_entity_mask_no = torch.tensor(d['non_entity_specific_mask_ids'])
            p = len(entity_mask_no)
            s += p
            entity_mask_no = entity_mask_no > 0
            entity_mask_no_list = torch.flatten(entity_mask_no.nonzero()).tolist()
            non_entity_mask_no = non_entity_mask_no > 0
            non_entity_mask_no_list = torch.flatten(non_entity_mask_no.nonzero()).tolist()
            entity_mask.append(entity_mask_no_list)
            non_entity_mask.append(non_entity_mask_no_list)
            non_mask_list = [i for i in range(len(entity_mask_no)) if
                             i not in entity_mask_no_list + non_entity_mask_no_list]
            non_masked.append(non_mask_list)
        return entity_mask, non_entity_mask, non_masked

    #calculate a weight for the arbitrary masked tokens (base level masking), entity masked tokens
    # (entity-level masking) as well as non-masked tokens
    def compute_weights(self, e_ms, ne_ms, n_ms):
        len_e_ms = len([i for ins in e_ms for i in ins])
        len_ne_ms = len([i for ins in ne_ms for i in ins])
        len_n_ms = len([i for ins in n_ms for i in ins])
        total_len = len_e_ms + len_ne_ms + len_n_ms
        self.weight_matrix = torch.zeros(3)
        for n,m in enumerate([len_e_ms, len_ne_ms, len_n_ms]):
            self.weight_matrix[n] = 1 - float(m/total_len)
        return self._softmax(self.weight_matrix)

    #compute a tensor with a weight for each token with respect to the mask (mask specific weight)
    def compute_mask_specific_weights(self, seq_len):
        assert len(self.entity_mask) == len(self.non_entity_mask) == len(self.non_masked), "Something is wrong"
        n = len(self.entity_mask)
        weights = torch.zeros(n, seq_len)
        for i in range(n):
            weights[i, self.entity_mask[i]] = self.computed_weights[0]
            weights[i, self.non_entity_mask[i]] = self.computed_weights[1]
            weights[i, self.non_masked[i]] = self.computed_weights[2]
        return weights

    #compute a cross entropy loss
    def compute_loss(self, inputs, labels, mask_specific_weights=True, **kwargs):
        inputs = inputs.get("logits")
        inputs = inputs.view(-1, inputs.size(-1))
        labels = labels.view(-1)
        train_loss = self._criterion(inputs, labels)
        return train_loss

