import torch
import torch.nn as nn


def fgsm(model, x, y, epsilon):
    """
    Generate adversarial examples using FGSM.

    Args:
        model: trained neural network
        x: input images, shape [B, C, H, W]
        y: true labels, shape [B]
        epsilon: perturbation budget

    Returns:
        x_adv: adversarial images
    """

    x = x.clone().detach()
    x.requires_grad_(True)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    outputs = model(x)
    loss = criterion(outputs, y)

    model.zero_grad()

    loss.backward()

    x_adv = x + epsilon * x.grad.sign()

    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()
