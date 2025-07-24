from collections.abc import Callable, Iterable 
from typing import Optional 
import torch 
import math

class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr} 
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                   continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value. 
                grad = p.grad.data # Get the gradient of loss with respect to p. Access data directly so autograd does not track it
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place. 
                state["t"] = t + 1 # Increment iteration number.
        
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps} 
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    state = self.state[p]
                    m = state.get("m", torch.zeros_like(p))
                    v = state.get("v", torch.zeros_like(p))
                    t = state.get("t", 1)

                    m = beta1 * m + (1 - beta1) * p.grad
                    v = beta2 * v + (1 - beta2) * p.grad.pow(2)

                    lr_t = lr * math.sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t))
                    
                    p -= lr_t * m / (torch.sqrt(v) + eps)
                    p -= lr * weight_decay * p

                    state["m"] = m
                    state["v"] = v
                    state["t"] = t + 1
        return loss

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it < cosine_cycle_iters:
        coef = 1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        return min_learning_rate + 0.5 * coef * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]

    # we need to calculate the norm for all parameters grad combined, not based on each parameter
    # take the norm of each parameter's norm has the same effect of directly taking the combined norm
    # sqrt(sqrt(a^2 + b^2)^2 + sqrt(c^2)^2) = sqrt(a^2 + b^2 + c^2)
    norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g) for g in grads]))
    if norm > max_l2_norm:
        coef = max_l2_norm / (norm + eps)
        for g in grads:
            g.data.mul_(coef)
    