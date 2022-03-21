import json
import os
import os.path as op

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, data_dir=r'data/ref', train=True, no_augment=True, aug_prob=0.5, img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225), tokenizer=None):
        super(ImageDataset, self).__init__()
        self.__dict__.update(locals())
        self.aug = train and not no_augment
        self.path_list = None
        if not train:
            self.tokenizer = tokenizer
        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value.

        if self.train:
            train_image_file_list_path = op.join(self.data_dir, 'train2017')
            self.path_list = [op.join(train_image_file_list_path, i) for i in os.listdir(train_image_file_list_path)]
        else:
            val_image_file_list_path = op.join(self.data_dir, 'val2017')
            self.path_list = []
            self.captions = []
            self.sentence = []
            self.annotations_dir = op.join(self.data_dir, 'annotations')
            with open(op.join(self.annotations_dir, 'captions_val2017.json'), 'r') as f:
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
                    self.sentence.append(caption)
                    self.captions.append(self.tokenizer(caption).squeeze())
                    self.path_list.append(op.join(val_image_file_list_path, file_name))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img = Image.open(path).convert('RGB')
        trans = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(self.aug_prob),
            transforms.RandomVerticalFlip(self.aug_prob),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ]) if self.train else transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])

        img_tensor = trans(img)

        if self.train:
            return img_tensor
        else:
            return img_tensor, self.captions[idx], self.sentence[idx]
