# coding: utf-8

import torch
from torch.nn import functional
from torch import nn


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss




def contrastive_loss(features,temperature=1.0):
    #hiddens: [batch,size]
    # Normalize the features
    features = nn.functional.normalize(features, dim=-1)

    # Compute the similarity matrix
    sim_matrix = torch.matmul(features, features.T)

    # Compute the loss
    pos_mask = torch.eye(features.size(0)).bool().to(features.device)
    neg_mask = ~pos_mask
    pos_sim = sim_matrix[pos_mask].view(features.size(0), 1)
    neg_sim = sim_matrix[neg_mask].view(features.size(0), -1)
    logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature
    labels = torch.zeros(features.size(0)).long().to(features.device)
    loss = nn.functional.cross_entropy(logits, labels)

    return loss


def masked_soft_cross_entropy_with_weight(logits, target, length,weights):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len,num_classes) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, num_classes)
    target_flat=target.view(-1,target.size(-1))
    target_flat =  functional.softmax(target_flat).float()
    # losses_flat: (batch * max_len, 1)
    losses_flat = -target_flat * log_probs_flat
    losses_flat = losses_flat.sum(dim = 1)

    # losses: (batch, max_len)
    losses = losses_flat.view(target.size(0),target.size(1))
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum(dim=1) / length.float()
    loss=loss*weights
    # loss=loss.sum()/weights.sum()
    loss=loss.mean()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss


def masked_soft_cross_entropy_batch(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len,num_classes) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, num_classes)
    target_flat=target.view(-1,target.size(-1))
    target_flat =  functional.softmax(target_flat).float()
    # losses_flat: (batch * max_len, 1)
    losses_flat = -target_flat * log_probs_flat
    losses_flat = losses_flat.sum(dim = 1)

    # losses: (batch, max_len)
    losses = losses_flat.view(target.size(0),target.size(1))
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum(dim=1) / length.float()

    # loss=loss.mean()

    return loss

def masked_soft_cross_entropy(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len,num_classes) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, num_classes)
    target_flat=target.view(-1,target.size(-1))
    target_flat =  functional.softmax(target_flat).float()
    # losses_flat: (batch * max_len, 1)
    losses_flat = -target_flat * log_probs_flat
    losses_flat = losses_flat.sum(dim = 1)

    # losses: (batch, max_len)
    losses = losses_flat.view(target.size(0),target.size(1))
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss

def masked_cross_entropy_without_logit(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))

    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.log(logits_flat + 1e-12)

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())

    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss

