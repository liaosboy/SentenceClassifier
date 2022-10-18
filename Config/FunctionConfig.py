from .Config import Config
class FunctionConfig(Config):
    def __init__(self) -> None:
        super(FunctionConfig).__init__()
        self.name = "Function"
        self.file_path = "D:/other/text-detection-lstm/storage/FunDataset.csv"
        self.MAX_LENGTH = 20
        self.num_class = 6
        self.label_to_index = {
            'test':-1,
            'account': 0,
            'budget': 1,
            'category': 2,
            'fixed': 3,
            'fixed_cate': 4,
            'income': 5,
        }
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.model_path ="D:/other/text-detection-lstm/ModelTrain/weight/Function_best.pt"