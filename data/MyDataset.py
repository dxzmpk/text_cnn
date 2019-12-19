from torch.utils.data import Dataset
import nltk
import random
from nltk.corpus import movie_reviews
import torch

class MyDataset(Dataset):

    def __init__(self, documents_vector, type, transform):
        super().__init__()
        if type == 'Train':
            self.documents = documents_vector[0:1800]
        else:
            self.documents = documents_vector[1800:-1]

    def __getitem__(self, idx):
        return torch.LongTensor(self.documents[idx]['vector']), self.documents[idx]['label']
 
    def __len__(self):
        return len(self.documents)


# Test
# ds = MyDataset([{"vector":[1,2,3], 'label':1} for i in range(2000)],type= 'Train', transform = None)
# item1 = ds[1]
# print(item1)