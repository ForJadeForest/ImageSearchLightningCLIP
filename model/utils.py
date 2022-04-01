import os

import numpy as np
import skimage
import torch
from PIL import Image
from clip import tokenize
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from data.test_image_dataset import TestImageDataset


def generate_figure(similarity, texts, images):
    count = len(texts)
    figure = plt.figure(figsize=(16, 16))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.colorbar()
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)
    return figure


def add_similarity(model, tb_write, epoch, **kwargs):
    descriptions = {
        "page": "a page of text about segmentation",
        "chelsea": "a facial photo of a tabby cat",
        "astronaut": "a portrait of an astronaut with the American flag",
        "rocket": "a rocket standing on a launchpad",
        "motorcycle_right": "a red motorcycle standing in a garage",
        "camera": "a person looking at a camera on a tripod",
        "horse": "a black-and-white silhouette of a horse",
        "coffee": "a cup of coffee on a saucer"
    }

    original_images = []
    images = []
    texts = []
    plt.figure(figsize=(16, 16))

    for filename in [filename for filename in os.listdir(skimage.data_dir) if
                     filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue
        image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
        original_images.append(image)
        images.append(model.teacher.preprocess(image))
        texts.append(descriptions[name])
    image_input = torch.tensor(np.stack(images), device=kwargs['device'])
    text_tokens = tokenize([desc for desc in texts]).to(kwargs['device'])

    if kwargs['model_name'] == 'model_text_distilled':
        text_features, tea_text_features = model(text_tokens)
        image_features = model.teacher.encode_image(image_input).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        tea_text_features = tea_text_features / tea_text_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
        tea_similarity = (image_features @ tea_text_features.T).softmax(dim=-1).cpu().numpy()
        similarity100 = (100 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
    else:
        image_features, tea_image_features = model(image_input)
        text_features = model.teacher.encode_text(text_tokens).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        tea_image_features = tea_image_features / tea_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features @ image_features.T).softmax(dim=-1).cpu().numpy()
        tea_similarity = (text_features @ tea_image_features.T).softmax(dim=-1).cpu().numpy()
        similarity100 = (100 * text_features @ image_features.T).softmax(dim=-1).cpu().numpy()
    tb_write.add_figure('Similarity/student', generate_figure(similarity, texts, original_images), epoch)
    tb_write.add_figure('Similarity/teacher', generate_figure(tea_similarity, texts, original_images), epoch)
    tb_write.add_figure('Similarity/100 * student', generate_figure(similarity100, texts, original_images), epoch)

    plt.clf()

    plt.close("all")


def imageQuery(query, model, encodes_path, file_path, device, tb_write, epoch):
    student = model.student
    teacher = model.teacher

    image_set = TestImageDataset(file_path, preprocess=teacher.preprocess)
    for file_names, images in DataLoader(image_set, batch_size=256, shuffle=False, num_workers=16):
        out = student(images.to(device))
        for file_name, feature in zip(file_names, out):
            save = feature / feature.norm(dim=0, keepdim=True)
            torch.save(save, os.path.join(encodes_path, file_name.split('.')[0] + '.pth'))

    encodes = []
    name = [s.split('.')[0] for s in os.listdir(encodes_path)]
    for file in os.listdir(encodes_path):
        res = torch.load(os.path.join(encodes_path, file))
        encodes.append(res)
    encodes = torch.stack(encodes, 0).to(device)
    query = tokenize(query)

    text_features = teacher.encode_text(query.to(device)).float()
    text_probs = (100 * text_features @ encodes.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(20, dim=-1)

    figure = plt.figure(figsize=(10, 8), dpi=200)

    top_labels = top_labels.squeeze(dim=0)
    for i, index in enumerate(top_labels):
        filename = name[index]
        path = os.path.join(file_path, filename + '.jpg')
        image = Image.open(path).convert("RGB")
        plt.subplot(5, 4, i + 1)
        plt.imshow(image)
        plt.axis("off")
    tb_write.add_figure('Similarity/teacher', figure, epoch)
    tb_write.add_figure('Similarity/100 * student', figure, epoch)
