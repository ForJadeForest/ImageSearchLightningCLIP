import os

from PIL import Image
from torch.utils.data import Dataset


class TestImageDataset(Dataset):
    def __init__(self, file_path, preprocess):
        super(TestImageDataset, self).__init__()
        self.file_path = file_path
        self.file_list = os.listdir(file_path)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        photo_path = os.path.join(self.file_path, self.file_list[item])
        image = Image.open(photo_path).convert('RGB')
        return self.file_list[item], self.preprocess(image)
