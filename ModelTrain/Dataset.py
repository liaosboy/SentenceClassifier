import torch
import torch.utils.data as Data
import numpy as np

class Dataset(Data.Dataset):
    def __init__(self, data_list, MAX_LENGTH, label_to_index, transform = None):
        self.transform = transform
        self.MAX_LENGTH = MAX_LENGTH
        self.total_size = len(data_list)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-chinese')
        self.label_to_index = label_to_index
        self.features, self.labels = zip(*data_list)
        self.features = self.tokenizer(list(self.features)).input_ids

    def __toTensor(self, x):
        x = np.array(x)
        x = torch.tensor(x)
        return x

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        f = self.features[idx]
        f += [0] * (self.MAX_LENGTH - len(f))
        l = self.label_to_index[self.labels[idx]]
        f = self.__toTensor(f)
        l = self.__toTensor(l)
        f = f.unsqueeze(1)

        return f, l