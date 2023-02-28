# CLIP Distill

## 0. Basic Idea
Considering that it is difficult for consumer-grade GPUs to handle two CLIP models simultaneously, this project separately distills the text and image encoders.

## 1. Prepare Your Data
⚠️ All data should be placed in the same directory.

### Images
You can prepare any image data (such as ImageNet, mscoco...) and then copy all the image data to a single folder.
- Tips: Vit models require a large amount of data to achieve good results (at least 1M).

However, this project will definitely use [MSCOCO2017Val](http://images.cocodataset.org/zips/val2014.zip) as validation data. So please download it in advance.

### Text
Two datasets are used for text data:
1. Annotations from [MSCOCO2017](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).
2. Captions from [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).

- After downloading, you should get these two files (put them in the COCO and CC folders respectively, and these two folders should be in the same directory!):
  1. /path/to/data/COCO/annotations/captions_train2017.json
  2. /path/to/data/CC/Train_GCC-training.tsv


Finally, you will get a data folder like this:
```
/path/to/data/image/...   # image folder contains only images, do not add additional folders!
/path/to/data/COCO/val2017/...
/path/to/data/COCO/annotations/...
/path/to/data/CC/Train_GCC-training.tsv
```


## 2. Train the Distillation Model

1. Multi-GPU Image Encoder
```
python main.py --data_dir /path/to/data/ --strategy=ddp --gpus=0,1,2,3 --precision=16 --max_epochs=100 --dataset=image_dataset --model_name=model_image_distilled
```
2. Multi-GPU Text Encoder
```
python main.py --data_dir /path/to/data/ --strategy=ddp --gpus=0,1,2,3 --precision=16 --max_epochs=100 --dataset=text_dataset --model_name=model_text_distilled
```

3. Single-GPU Image Encoder
```
python main.py --data_dir /path/to/data/ --gpus=0 --precision=16 --max_epochs=100 --dataset=image_dataset --model_name=model_image_distilled
```

4. Single-GPU Text Encoder
```
python main.py --data_dir /path/to/data/ --gpus=0 --precision=16 --max_epochs=100 --dataset=text_dataset --model_name=model_text_distilled
```

For more PyTorch-Lightning training parameters, please refer to https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api

### Loss Setting
The project sets up 3 loss functions, including l1, ce, and kl.

If [l1, ce, kl] are selected, the calculation of the final loss is as follows:
$$
loss = weight_1 * scale_1 * l1 + weight_2 * scale_2 * ce  + weight_3 * scale_3 * kl
$$
Tips: The order of the loss list is consistent with the order of weight and scale.

### Student Model Setting
```
    # Vit Model Hyperparameters
    parser.add_argument('--input_resolution', default=224, type=int)
    parser.add_argument('--patch_size', default=32, type=int)
    parser.add_argument('--width', default=576, type=int)
    parser.add_argument('--layers', default=6, type=int)
    parser.add_argument('--heads', default=24, type=int)

    # Language Transformer Model Hyperparameters
    parser.add_argument('--context_length', default=77, type=int)
    parser.add_argument('--vocab_size', default=49408, type=int)
    parser.add_argument('--transformer_width', default=128, type=int)
    parser.add_argument('--transformer_layers', default=6, type=int)
    parser.add_argument('--transformer_heads', default=8, type=int)
```
These parameters can be modified to adjust the structure of the Student model.