import torch
from torch import nn

try:
    from .teacher_model import TeacherModel
    from .transformer_student import TransformerStudent
except:
    from teacher_model import TeacherModel
    from transformer_student import TransformerStudent


class ModelTextDistilled(nn.Module):
    def __init__(self, teacher_name, context_length, vocab_size, transformer_width, transformer_layers,
                 transformer_heads, output_dim):
        super(ModelTextDistilled, self).__init__()
        self.teacher = TeacherModel(teacher_name=teacher_name)
        for p in self.parameters():
            p.requires_grad = False

        self.student = TransformerStudent(context_length, vocab_size, transformer_width, transformer_layers,
                                          transformer_heads, output_dim)

    def forward(self, text):
        stu_encode = self.student(text)
        with torch.no_grad():
            tea_encode = self.teacher.encode_text(text).float()
        return stu_encode, tea_encode


if __name__ == '__main__':
    text_model = ModelTextDistilled('ViT-B/32', 77, 49408, 128, 6, 8, 512).to('cuda:0')
    print(text_model(torch.randint(low=0, high=49409, size=(3, 77)).cuda())[0].shape)
