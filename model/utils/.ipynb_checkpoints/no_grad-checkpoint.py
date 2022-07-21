import torch

def without_grad(f):
    def new_f(*arg, **kwargs):
        with torch.no_grad():
            return f(*arg, **kwargs)
    return new_f
