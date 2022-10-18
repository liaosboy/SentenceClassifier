from torch.autograd import Variable
from ModelTrain.Model import Model
from ModelTrain.Dataset import Dataset
from Config.Config import Config
from Config.CategoryConfig import CategoryConfig
from Config.FunctionConfig import FunctionConfig
import torch

class Predictions:
    def __init__(self) -> None:
        self.config:Config
    
    def __set_config(self, config:Config):
        self.config = config

    def __make_data(self, sentence):
        dataset = Dataset([[sentence, 'test']], self.config.MAX_LENGTH, self.config.label_to_index)
        return dataset[0]

    def __get_model(self, chose):  # 0:is Function Detect, other is Category Detect
        if chose == '0':
            self.__set_config(FunctionConfig())
        else:
            self.__set_config(CategoryConfig())

        model:Model = Model(self.config.input_size, self.config.MAX_LENGTH, self.config.hidden_size, self.config.num_layers, self.config.num_class)
        model.load_state_dict(torch.load(self.config.model_path))
        model.cuda()
        model.eval()
        return model

    def run(self, sentence, chose):
        model = self.__get_model(chose)
        input = self.__make_data(sentence)
        input = Variable(input[0].type(torch.cuda.FloatTensor)).unsqueeze(0)
        output = model(input)
        _, pred = torch.max(output.data, 1)
        pred = pred.cpu()
        return self.config.index_to_label[pred.item()]


if __name__ == '__main__':
    pred = Predictions()
    text = input("請輸入要判斷的句子:").lower()
    while not text == 'q':
        result = pred.run(text, '0')
        print(f"\n判斷結果為:  {result}")
        print("--------------")
        text = input("請輸入要判斷的句子:")
