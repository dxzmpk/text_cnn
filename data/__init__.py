import numpy as np
from .MyDataset import MyDataset
from torch.utils.data import DataLoader, random_split
from logger import logging
from torchvision.datasets.folder import ImageFolder

def get_dataloaders(
        document_vectors,
        train_transform=None,
        val_transform=None,
        split=(0.5, 0.5),
        batch_size=32):
    """
    This function returns the train, val and test dataloaders.
    """
    # create the datasets
    train_ds = MyDataset(document_vectors, 'Train', train_transform)
    val_ds = MyDataset(document_vectors, 'Test', val_transform)
    # now we want to split the val_ds in validation and test
    lengths = np.array(split) * len(val_ds)
    lengths = lengths.astype(int)
    left = len(val_ds) - lengths.sum()
    # we need to add the different due to float approx to int
    lengths[-1] += left

    val_ds, test_ds = random_split(val_ds, lengths.tolist())
    logging.info(f'Train samples={len(train_ds)}, Validation samples={len(val_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dl, val_dl, test_dl
