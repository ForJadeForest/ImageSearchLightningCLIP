import numpy as np
import torch
from torch import nn

from .common import VisionTransformer


class VitStudent(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super(VitStudent, self).__init__()

        self.student_model = VisionTransformer(input_resolution=input_resolution,
                                               patch_size=patch_size,
                                               width=width,
                                               layers=layers,
                                               heads=heads,
                                               output_dim=output_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))

    def encode_image(self, image):
        return self.student_model(image)

    def forward(self, image, text_feature=None):
        if text_feature is None:
            return self.student_model(image)
        image_feature = self.student_model(image)

        logits_per_image = self.calculate_logits(image_feature, text_feature)
        return logits_per_image

    def calculate_logits(self, image_feature, text_feature):
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_feature @ text_feature.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image
