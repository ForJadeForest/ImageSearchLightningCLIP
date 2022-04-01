import torch
from torch import nn

try:
    from .common import Transformer, LayerNorm
except:
    from common import Transformer, LayerNorm


class TransformerStudent(nn.Module):
    def __init__(self, context_length, vocab_size, transformer_width, transformer_layers, transformer_heads,
                 output_dim):
        super(TransformerStudent, self).__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads,
                                       self.build_attention_mask())
        self.ln_final = LayerNorm(transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, output_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, image_feature=None):
        if image_feature is None:
            return self.encode_text(text)
        text_feature = self.encode_text(text)

        logits_per_text = self.calculate_logits(image_feature, text_feature)
        return logits_per_text

    def calculate_logits(self, image_feature, text_feature):
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_feature @ text_feature.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image.t()


if __name__ == '__main__':
    text_model = TransformerStudent(77, 49408, 128, 6, 8, 512)
    print(text_model(torch.randint(low=0, high=49409, size=(3, 77))).shape)
