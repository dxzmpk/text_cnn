import torch
import numpy as np
from torchvision.utils import make_grid

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def show_dl(dl, n=6):
    batch = None
    for batch in dl:
        break
    vectors = batch[0][n]
    print(type(vectors))
    print(vectors)
    
