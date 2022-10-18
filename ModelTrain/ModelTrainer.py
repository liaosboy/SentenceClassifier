from torch.autograd import Variable
from Dataset import Dataset
from Model import Model
from Config.Config import Config
from Config.CategoryConfig import CategoryConfig
from Config.FunctionConfig import FunctionConfig

import torch.utils.data as Data
import pandas as pd
import torch.optim as opt
import torch.nn as nn

import torch
import random


class ModelTrainer():
    def __init__(self, config:Config) -> None:
        self.file_path = config.file_path
        self.MAX_LENGTH = config.MAX_LENGTH
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_class = config.num_class
        self.train_dataset, self.test_dataset = self.__load_data(self.file_path, config.label_to_index)
        self.model = Model(self.input_size, self.MAX_LENGTH, self.hidden_size, self.num_layers, self.num_class)
        self.model.cuda()

    def __load_data(self, file_path, l2i):
        csv = pd.read_csv(file_path)
        df = pd.DataFrame(csv)
        feature = df['title'].tolist()
        label = df['label'].tolist()
        data = list(zip(feature, label))
        random.shuffle(data)
        idx_ = int(len(data) * 0.9)
        data_train = data[:idx_]
        data_test = data[idx_:]
        train_dataset = Dataset(data_train, self.MAX_LENGTH, l2i)
        test_dataset = Dataset(data_test, self.MAX_LENGTH, l2i)
        return train_dataset, test_dataset

    def train(self):
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = opt.Adam(self.model.parameters(), lr=0.0007)
        
        train_loader = Data.DataLoader(
            dataset = self.train_dataset,
            batch_size = 200,
            shuffle=True
        )
        best = 0
        for epoch in range(10000):
            for i, (seqs, labels) in enumerate(train_loader):
                seqs = Variable(seqs.type(torch.cuda.FloatTensor))
                labels = Variable(labels.type(torch.cuda.LongTensor))
                optimizer.zero_grad()
                outputs = self.model(seqs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            if (epoch) % 500 == 0:
                score = self.test()
                print(f"Epoch:{epoch+1}, Loss:{'%.4f'%loss}")
                if score > best:
                    best = score
                    torch.save(self.model.state_dict(), f"weight/{config.name}_best.pt")

    def test(self):
        test_loader = Data.DataLoader(
            dataset = self.test_dataset,
            batch_size = 200,
            shuffle=False
        )

        correct = 0
        total = 0
        for seqs, labels in test_loader:
            seqs = Variable(seqs.type(torch.cuda.FloatTensor))
            output = self.model(seqs)
            _, preds = torch.max(output.data, 1)
            preds = preds.cpu()
            preducted = preds.eq(labels)
            total += labels.size(0)
            correct += (preducted.int()).sum().item()
        
        print(f"Acc:{round((100.0 * float(correct) / float(total)),3)}%")
        return (100.0 * float(correct) / float(total))

if __name__ == "__main__":
    config = FunctionConfig()
    model = ModelTrainer(config)
    model.train()
    model.test()