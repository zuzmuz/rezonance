import torch

def run(*args, **kwargs):
    print(f"Default device: {torch.get_default_device()}")
