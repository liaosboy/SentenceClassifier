from .Config import Config
class CategoryConfig(Config):
    def __init__(self) -> None:
        super(CategoryConfig).__init__()
        self.name = "Category"
        self.file_path = "D:/other/text-detection-lstm/storage/CateDataset.csv"
        self.MAX_LENGTH = 15
        self.num_class = 16
        self.label_to_index = {
            'test':-1,
            '飲食': 0,
            '服飾': 1,
            '居家': 2,
            '交通': 3,
            '教育': 4,
            '交際娛樂': 5,
            '交流通訊': 6,
            '醫療保健': 7,
            '生活用品': 8,
            '金融保險': 9,
            '美容美髮': 10,
            '運動用品': 11,
            '3C產品': 12,
            '稅金/日常費用': 13,
            '寵物百貨': 14,
            '其他支出': 15
        }
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.model_path = "D:/other/text-detection-lstm/ModelTrain/weight/Category_best.pt"