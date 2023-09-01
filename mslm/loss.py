from abc import ABC
import torch

class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass

class MslmLoss(Loss):
    def __init__(self, batch_size, seq_len, vocab_size, *args, **kwargs):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self._softmax = torch.nn.Softmax(dim=0)

    #compute a cross entropy loss
    def compute_loss(self, inputs, labels, ms_weights, weight_matrix, reduction='none', **kwargs):
        """
        :param inputs: logits from a mask language model
        :param labels: original input ids of the input batch
        :param ms_weights: matrix with similar dimensions to input batch, however weight values in the masked positions
        :param weight_matrix: tensor([w1,w2,w3]) a tensor with weights for different mask types,
        i.e. w1 - entity-level mask weights, w2 - base-level or random mask weight, w3 - non-masked weights
        :param reduction: 'mean', compute the average loss across samples within the batch otherwise, 'sum' sum it
        :return: loss
        """
        inputs = inputs.get("logits")
        loss = self._criterion(inputs.view(-1, self.vocab_size), labels.view(-1))

        if reduction == 'mean':
            loss_per_pos = loss.view(inputs.size(0), inputs.size(1))
            ms_weights_entity_mask_pos = torch.stack([(ms_weights == w).float() for w in weight_matrix[:2]])
            ms_weights_entity_mask_pos = ms_weights_entity_mask_pos.sum(axis=0)
            ms_weights = ms_weights_entity_mask_pos.view(-1, ms_weights_entity_mask_pos.size(-1)) + ms_weights
            loss_per_sample = torch.mul(loss_per_pos, ms_weights).mean(axis=1)
            weights = 1 + ms_weights_entity_mask_pos.sum(axis=1)
            weighted_mask_specific_loss = loss_per_sample * weights
            return weighted_mask_specific_loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()

class DetectionLoss(Loss):
    def __init__(self, num_labels, *args, **kwargs):
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_labels = num_labels

    #compute a cross entropy loss
    def compute_loss(self, inputs, labels, ms_weights, weight_matrix, reduction='none', **kwargs):
        """
        :param inputs: logits from a mask language model
        :param labels: original input ids of the input batch
        :param ms_weights: matrix with similar dimensions to input batch, however weight values in the masked positions
        :param weight_matrix: tensor([w1,w2,w3]) a tensor with weights for different mask types,
        i.e. w1 - entity-level mask weights, w2 - base-level or random mask weight, w3 - non-masked weights
        :param reduction: 'mean', compute the average loss across samples within the batch otherwise, 'sum' sum it
        :return: loss
        """
        loss = self._criterion(inputs.view(-1, self.num_labels), labels.view(-1))
        if reduction == 'mean':
            loss_per_pos = loss.view(inputs.size(0), inputs.size(1))
            ms_weights_entity_mask_pos = torch.stack([(ms_weights == w).float() for w in weight_matrix[:2]])
            ms_weights_entity_mask_pos = ms_weights_entity_mask_pos.sum(axis=0)
            ms_weights = ms_weights_entity_mask_pos.view(-1, ms_weights_entity_mask_pos.size(-1)) + ms_weights
            loss_per_sample = torch.mul(loss_per_pos, ms_weights).mean(axis=1)
            weights = 1 + ms_weights_entity_mask_pos.sum(axis=1)
            weighted_mask_specific_loss = loss_per_sample * weights
            return weighted_mask_specific_loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()