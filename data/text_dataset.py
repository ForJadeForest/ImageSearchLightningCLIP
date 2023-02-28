import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, cache_dir, data_dir=r'data/ref', train=True, overwrite=False,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):
        super(TextDataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.train = train
        self.tokenizer = tokenize
        if self.train:
            self.tokenize_text = self.load(overwrite)
        if not self.train:
            self.img_mean = img_mean
            self.img_std = img_std
            self.sentences, self.captions, self.path_list = self.load(overwrite)

    def process(self):
        raw_text = []
        if self.train:
            coco2017_file = self.data_dir / 'COCO' / 'annotations' / 'captions_train2017.json'
            cc_file = self.data_dir / 'CC' / 'Train_GCC-training.tsv'

            with cc_file.open('r', encoding='utf8') as f:
                for content in f.readlines():
                    raw_text.append(content.split('\t')[0])
            with coco2017_file.open('r', encoding='utf8') as f:
                res = json.load(f)
                for annotation in res['annotations']:
                    raw_text.append(annotation['caption'])

            print('All data: {} Begin tokenizing...'.format(len(raw_text)))
            tokenize_text = []
            for text in tqdm(raw_text):
                tokenize_text.append(self.tokenizer(text).squeeze(), truncate=True)

            return torch.stack(tokenize_text)
        else:
            val_image_file_list_path = self.data_dir / 'COCO' / 'val2017'
            path_list = []
            captions = []
            sentences = []
            file_dir = self.data_dir / 'COCO' / 'annotations' / 'captions_val2017.json'
            with file_dir.open('r', encoding='utf8') as f:
                data = json.load(f)
            images = data['images']
            id2caption = {}
            id2filename = {}
            for image in images:
                id2filename[image['id']] = image['file_name']
            for annotation in data['annotations']:
                id2caption[annotation['image_id']] = annotation['caption']
            for id, file_name in id2filename.items():
                caption = id2caption.get(id, None)
                if caption:
                    sentences.append(caption)
                    captions.append(self.tokenizer(caption).squeeze())
                    path_list.append(val_image_file_list_path / file_name)

            return sentences, captions, path_list

    def load(self, overwirite):
        cache_path = self.cache_dir / 'cache-train.pth' if self.train else self.cache_dir / 'cache-val.pth'
        if overwirite or not cache_path.exists():
            print('重写/不存在缓存文件，开始处理文件')
            if self.train:
                tokenize_text = self.process()
                torch.save({'data_set': tokenize_text}, cache_path)
                return tokenize_text
            else:
                sentences, captions, path_list = self.process()
                torch.save({
                    'data_set': [
                        sentences,
                        captions,
                        path_list
                    ]
                }, cache_path)
                return sentences, captions, path_list
        else:
            print('直接加载缓存文件')
            data = torch.load(cache_path)['data_set']
            print('加载完成！')
            return data

    def __len__(self):
        if self.train:
            return len(self.tokenize_text)
        else:
            return len(self.path_list)

    def __getitem__(self, idx):
        if self.train:
            return self.tokenize_text[idx]
        path = self.path_list[idx]
        img = Image.open(path).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        img_tensor = trans(img)
        return img_tensor, self.captions[idx], self.sentences[idx]


if __name__ == '__main__':
    from clip import tokenize
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    text_data = TextDataset('/home/pyz/data/cache', '/home/pyz/data', True, tokenize, True)
    for data in DataLoader(text_data, batch_size=2):
        print(data)
        break
    text_data = TextDataset('/home/pyz/data/cache', '/home/pyz/data', False, tokenize, False)
    for data in DataLoader(text_data, batch_size=1, shuffle=True):
        image, caption, sentence = data
        plt.imshow(image.squeeze(0).permute(1, 2, 0))
        print(caption, sentence)
        break
