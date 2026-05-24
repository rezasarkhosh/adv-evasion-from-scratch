import torch
import torch.nn as nn
import torch.optim as optim


def cw_linf_attack(
    model,
    x,
    y,
    tau=0.3,
    tau_decay=0.9,
    c=1.0,
    lr=0.01,
    num_steps=100,
    num_binary_steps=8,
    kappa=0.0,
):
    """
    Carlini & Wagner L-infinity attack, untargeted version.

    This implementation follows the same functional style as fgsm.py, pgd.py,
    and cw_l2.py.

    Args:
        model: trained classifier
        x: clean images in [0, 1], shape [B, C, H, W]
        y: true labels, shape [B]
        tau: initial L-infinity threshold
        tau_decay: shrink factor for tau after successful attack
        c: classification-loss weight
        lr: Adam learning rate
        num_steps: optimization steps per tau value
        num_binary_steps: number of tau-shrinking rounds
        kappa: confidence margin

    Returns:
        best_adv: best adversarial examples found
        best_linf: corresponding L-infinity distances
    """

    model.eval()

    device = x.device
    B = x.size(0)

    best_adv = x.clone().detach()
    best_linf = torch.full((B,), float("inf"), device=device)

    current_tau = tau

    for binary_step in range(num_binary_steps):
        delta = torch.zeros_like(x, device=device, requires_grad=True)
        optimizer = optim.Adam([delta], lr=lr)

        for step in range(num_steps):
            x_adv = torch.clamp(x + delta, 0, 1)

            logits = model(x_adv)

            true_logit = logits.gather(1, y.unsqueeze(1)).squeeze(1)

            other_logits = logits.clone()
            other_logits.scatter_(1, y.unsqueeze(1), float("-inf"))
            max_other_logit = other_logits.max(dim=1)[0]

            f_value = torch.clamp(
                true_logit - max_other_logit,
                min=-kappa
            )

            # Penalize any perturbation above current_tau.
            # This pushes ||delta||_inf below tau while still forcing misclassification.
            excess = torch.clamp(delta.abs() - current_tau, min=0.0)
            linf_penalty = excess.view(B, -1).sum(dim=1)

            loss = (c * f_value + linf_penalty).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.clamp_(-1.0, 1.0)

        with torch.no_grad():
            x_adv_now = torch.clamp(x + delta, 0, 1)
            preds_now = model(x_adv_now).argmax(dim=1)

            success = preds_now != y

            linf_now = (
                (x_adv_now - x)
                .view(B, -1)
                .abs()
                .max(dim=1)[0]
            )

            better_attack = success & (linf_now < best_linf)

            best_linf[better_attack] = linf_now[better_attack]
            best_adv[better_attack] = x_adv_now[better_attack].detach()

            successes = success.sum().item()

            print(
                f"tau step {binary_step + 1}/{num_binary_steps} | "
                f"tau={current_tau:.5f} | "
                f"successes={successes}/{B} | "
                f"best_found={(best_linf < float('inf')).sum().item()}/{B}"
            )

        current_tau *= tau_decay

    return best_adv.detach(), best_linf.detach()