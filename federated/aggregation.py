import torch

@torch.no_grad()
def fedavg(state_dicts, weights, cast_back=True):
    if len(state_dicts) == 0:
        return {}

    if len(weights) != len(state_dicts):
        raise ValueError("len(weights) must equal len(state_dicts)")

    weights = [float(w) for w in weights]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(weights)
        total_weight = float(len(weights))

    keys = state_dicts[0].keys()
    for sd in state_dicts[1:]:
        if sd.keys() != keys:
            raise ValueError("State dict keys mismatch across clients")

    averaged = {}
    for k in keys:
        t0 = state_dicts[0][k]

        if torch.is_floating_point(t0):
            acc = torch.zeros_like(t0.detach().cpu(), dtype=torch.float32)
            for sd, w in zip(state_dicts, weights):
                acc += sd[k].detach().cpu().to(torch.float32) * w
            out = acc / total_weight
            if cast_back:
                out = out.to(dtype=t0.dtype)
            averaged[k] = out
        else:
            averaged[k] = t0.detach().cpu()

    return averaged
