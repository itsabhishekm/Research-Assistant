
import torch

if not hasattr(torch, 'classes'):
    torch.classes = type('classes', (), {})()
else:
    torch.classes.__path__ = []  

from ui import run_app

if __name__ == "__main__":
    run_app()
