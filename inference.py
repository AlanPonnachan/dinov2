from dinov2.models.vision_transformer import vit_base
import torch
from PIL import Image
import requests
from functools import partial
from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

#load model
model = vit_base(
    patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=0,
)

#equip the model with weights
state_dict = torch.hub.load_state_dict_from_url(
    'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
    map_location='cpu')
model.load_state_dict(state_dict)

#load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

#preprocess image
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

pixel_values = transformations(image).unsqueeze(0)

output = model.forward_features(pixel_values)

print(output)




