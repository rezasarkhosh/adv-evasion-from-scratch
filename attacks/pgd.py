import torch
import torch.nn as nn



def pgd(model, x, y, epsilon, alpha, num_steps, random_start=True):
    """
    Projected Gradient Descent attack.

    Args:
        model: trained neural network
        x: clean input images [B, C, H, W]
        y: true labels [B]
        epsilon: maximum perturbation size
        alpha: step size
        num_steps: number of PGD iterations
        random_start: whether to start from random noise inside epsilon-ball

    Returns:
        x_adv: adversarial images
    """

    model.eval()

    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        delta = torch.clamp(x + delta, 0, 1) - x
    else:
        delta = torch.zeros_like(x)

    delta.requires_grad_(True)

    for _ in range(num_steps):
        outputs = model(x + delta)
        loss = nn.CrossEntropyLoss()(outputs, y)

        grad = torch.autograd.grad(loss, delta)[0]

        delta = delta.detach() + alpha * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = torch.clamp(x + delta, 0, 1) - x
        delta.requires_grad_(True)

    x_adv = torch.clamp(x + delta, 0, 1)
    return x_adv.detach()