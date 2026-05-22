import torch
import torch.nn as nn
import torch.optim as optim


def cw_l2_attack(
    model,
    x,                  # (B, C, H, W), pixel values in [0, 1]
    y,                  # (B,) true labels
    c=1.0,              # trade-off between perturbation size and attack success
    lr=0.01,
    num_steps=1000,
    kappa=0.0,          # confidence margin
):
    """
    Carlini & Wagner L2 Attack (Untargeted)

    Paper:
    Carlini & Wagner,
    "Towards Evaluating the Robustness of Neural Networks"
    IEEE S&P 2017

    Args:
        model: trained classifier
        x: clean input images
        y: true labels
        c: trade-off constant
        lr: Adam learning rate
        num_steps: optimization iterations
        kappa: confidence margin

    Returns:
        best_adv: best adversarial examples found
        best_l2: corresponding L2 distances
    """

    model.eval()

    device = x.device
    B = x.size(0)

    # ------------------------------------------------------------
    # STEP 1 — Change of variable using tanh
    #
    # x_adv = (tanh(w) + 1) / 2
    #
    # This guarantees valid pixels in [0,1]
    # without hard clipping.
    # ------------------------------------------------------------

    eps_boundary = 1e-6

    x_clipped = torch.clamp(
        x,
        eps_boundary,
        1.0 - eps_boundary
    )

    w = torch.atanh(2 * x_clipped - 1).detach()
    w.requires_grad_(True)

    optimizer = optim.Adam([w], lr=lr)

    # Track best successful adversarial example
    best_adv = x.clone().detach()

    best_l2 = torch.full(
        (B,),
        float("inf"),
        device=device
    )

    # ------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------

    for step in range(num_steps):

        # Recover adversarial image from w
        x_adv = (torch.tanh(w) + 1) / 2

        delta = x_adv - x

        # --------------------------------------------------------
        # STEP 2 — Compute logits
        # --------------------------------------------------------

        logits = model(x_adv)

        # True-class logit
        true_logit = logits.gather(
            1,
            y.unsqueeze(1)
        ).squeeze(1)

        # Max other-class logit
        other_logits = logits.clone()

        other_logits.scatter_(
            1,
            y.unsqueeze(1),
            float("-inf")
        )

        max_other_logit = other_logits.max(dim=1)[0]

        # --------------------------------------------------------
        # STEP 3 — C&W objective
        #
        # f(x') = max(
        #     Z(x')_y - max_{i≠y} Z(x')_i,
        #     -kappa
        # )
        # --------------------------------------------------------

        f_value = torch.clamp(
            true_logit - max_other_logit,
            min=-kappa
        )

        # --------------------------------------------------------
        # STEP 4 — L2 loss
        #
        # ||delta||² + c * f(x')
        # --------------------------------------------------------

        l2_per_sample = (
            delta.view(B, -1)
            .pow(2)
            .sum(dim=1)
        )

        loss = (
            l2_per_sample + c * f_value
        ).sum()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # --------------------------------------------------------
        # STEP 5 — Track best successful attacks
        # --------------------------------------------------------

        with torch.no_grad():

            x_adv_now = (torch.tanh(w) + 1) / 2

            preds_now = model(x_adv_now).argmax(dim=1)

            success = (preds_now != y)

            l2_now = (
                (x_adv_now - x)
                .view(B, -1)
                .pow(2)
                .sum(dim=1)
                .sqrt()
            )

            better_attack = (
                success &
                (l2_now < best_l2)
            )

            best_l2[better_attack] = l2_now[better_attack]

            best_adv[better_attack] = (
                x_adv_now[better_attack]
                .detach()
            )

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------

        if step % 100 == 0:

            with torch.no_grad():

                finite = best_l2 < float("inf")

                if finite.any():
                    mean_best = best_l2[finite].mean().item()
                else:
                    mean_best = float("nan")

                print(
                    f"step {step:4d} | "
                    f"loss={loss.item():.4f} | "
                    f"successes={finite.sum().item()}/{B} | "
                    f"mean_best_l2={mean_best:.4f}"
                )

    return best_adv.detach(), best_l2.detach()