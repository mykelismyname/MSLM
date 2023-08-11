from abc import ABC
import torch

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

class MslmLoss(Loss):
    def __init__(self, dataset, *args, **kwargs):
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self._softmax = torch.nn.Softmax(dim=0)
        tracked_dataset_ids = self.compute_mask_specific_weights(dataset)
        self.entity_mask = [id for ins in tracked_dataset_ids[0] for id in ins]
        self.non_entity_mask = [id for ins in tracked_dataset_ids[1] for id in ins]
        self.non_masked = tracked_dataset_ids[2]
        self.computed_weights = self.compute_weights(self.entity_mask,
                                                 self.non_entity_mask,
                                                 self.non_masked)
        

    def compute_mask_specific_weights(self, dataset):
        entity_mask = []
        non_entity_mask = []
        non_masked = []
        for d in dataset:
            entity_mask_no = torch.tensor(d['entity_specific_mask_ids'])
            non_entity_mask_no = torch.tensor(d['non_entity_specific_mask_ids'])
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

    def compute_weights(self, e_ms, ne_ms, n_ms):
        len_e_ms = len(e_ms)
        len_ne_ms = len(ne_ms)
        len_n_ms = len(n_ms)
        total_len = len_e_ms + len_ne_ms + len_n_ms
        self.weight_matrix = torch.zeros(3)
        for n,m in enuermate([len_e_ms, len_ne_ms, len_n_ms]):
            self.weight_matrix[n] = 1 - float(m/total_len)
        return self._softmax(self.weight_matrix)

    def compute(self, inputs, labels, **kwargs):
        inputs = inputs.get("logits")
        inputs = inputs.view(-1, inputs.size(-1))
        labels = labels.view(-1)
        train_loss = self._criterion(inputs, labels)
        return train_loss
