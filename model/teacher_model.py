import clip
from torch import nn


class TeacherModel(nn.Module):
    def __init__(self, teacher_name):
        super().__init__()
        self.teacher_model, self.preprocess = clip.load(teacher_name)

    def encode_image(self, image):
        return self.teacher_model.encode_image(image)

    def encode_text(self, text):
        return self.teacher_model.encode_text(text)

    def forward(self, image, text):
        re_image_features = self.encode_image(image)
        re_text_features = self.encode_text(text)

        image_features = re_image_features / re_image_features.norm(dim=-1, keepdim=True)
        text_features = re_text_features / re_text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.teacher_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return re_image_features, re_text_features, logits_per_image


if __name__ == '__main__':
    teacher = TeacherModel("")
