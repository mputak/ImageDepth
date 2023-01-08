import torch
import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
   "stabilityai/stable-diffusion-2-depth",
   torch_dtype=torch.float16,
).to("cuda")
pipe.enable_attention_slicing() 

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# init_image = Image.open(requests.get(url, stream=True).raw)
init_image = Image.open("/home/markoputak/Downloads/ja.jpeg")
init_image = init_image.resize((480, 580))



prompt = "evil"
n_propmt = "bad, deformed, ugly, bad anotomy"


image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.7).images[0]

init_image.show()
image.show()