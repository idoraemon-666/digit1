import torch

from digit_writing.phase_normalized_loss import (
    detached_position_metrics,
    phase_normalized_l1,
    position_l1_metrics,
)

def l1_dist(x, y):
    """L1 loss"""
    return torch.mean(torch.sum(torch.abs(x - y), dim=-1))

def l1_weight(rnn, scale):
    l1 = 0
    for name, param in rnn.named_parameters():
        l1 += torch.mean(torch.abs(torch.flatten(param)))
    l1 *= scale
    return l1

def l1_rate(act, scale):
    l1 = scale * torch.mean(torch.abs(torch.flatten(act)))
    return l1

def l1_muscle_act(act, scale):
    l1 = scale * torch.mean(torch.sum(torch.abs(act), dim=-1))
    return l1

def simple_dynamics(act, mrnn, weight=1e-2):

    W_rec, W_rec_mask, W_rec_sign = mrnn.gen_w(mrnn.region_dict)
    if mrnn.constrained:
        W_rec = mrnn.apply_dales_law(W_rec, W_rec_mask, W_rec_sign)

    if mrnn.activation_name == "softplus":
        derivative = 1 / (1 + torch.exp(-act))
    elif mrnn.activation_name == "relu":
        derivative = torch.where(act > 0, 1., 0.)
    else:
        raise ValueError("Not implemented for activation")

    d_act = torch.mean(derivative, dim=(1, 0))

    update = W_rec * d_act**2
    update = weight * torch.norm(update)

    return update
