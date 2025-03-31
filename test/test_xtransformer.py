import torch
from x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 12,
        heads = 8
    )
).cuda()

x = torch.randint(0, 256, (1, 1024)).cuda()

y = model(x) # (1, 1024, 20000)
print(y.shape)