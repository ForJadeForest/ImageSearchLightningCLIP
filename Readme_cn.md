# CLIP Distill

## 0. 基本思路
考虑到消费型GPU很难负载两个CLIP模型，因此本项目是**分开**蒸馏文本，图像编码器。


## 1. Prepare your data
⚠️ 所有的数据放置在同一个目录下

### 图像
你可以准备任意的图像数据（比如 ImageNet，mscoco...）然后将所有图像数据复制到一个文件夹下。
- Tips：对于Vit模型需要大量的数据才能有较好的效果（至少1M）

但本项目一定会使用[MSCOCO2017Val](http://images.cocodataset.org/zips/val2014.zip)作为验证。所以请提前下载好。


### 文本
文本数据使用两个数据集
1. [MSCOCO2017](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)的annotation
2. [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/)中的Caption

- 下载好后应该得到这两个文件（分别放到COCO和CC文件夹下，这两个文件夹需要在同一个目录下！）
  1. /path/to/data/COCO/annotations/captions_train2017.json
  2. /path/to/data/CC/Train_GCC-training.tsv


最终，你会得到一个这样的数据文件夹
```
/path/to/data/image/...   # image内全是图像，不能添加额外的文件夹！
/path/to/data/COCO/val2017/...
/path/to/data/COCO/annotations/...
/path/to/data/CC/Train_GCC-training.tsv
```

## 2. train the distillation model
1. 多卡 image encoder
```
python main.py --data_dir /path/to/data/ --strategy=ddp --gpus=0,1,2,3 --precision=16 --max_epochs=100 --dataset=image_dataset --model_name=model_image_distilled
```
2. 多卡 text encoder
```
python main.py --data_dir /path/to/data/ --strategy=ddp --gpus=0,1,2,3 --precision=16 --max_epochs=100 --dataset=text_dataset --model_name=model_text_distilled
```

1. 单卡 image encoder
```
python main.py --data_dir /path/to/data/ --gpus=0 --precision=16 --max_epochs=100 --dataset=image_dataset --model_name=model_image_distilled
```
2. 单卡 text encoder
```
python main.py --data_dir /path/to/data/ --gpus=0 --precision=16 --max_epochs=100 --dataset=text_dataset --model_name=model_text_distilled
```

更多Pytorch-lightning训练参数可以参考 https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api

### Loss Setting
项目设置了3个loss，分别是l1, ce, kl

如果选择了[l1, ce, kl]
最终loss的计算方式为：
$$
loss = weight_1 * scale_1 * l1 + weight_2 * scale_2 * ce  + weight_3 * scale_3 * kl
$$
Tips：loss列表中的顺序和weight，scale的顺序是一致的

### Student模型设置
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
这些参数可以修改Student模型结构


