# from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
# from torch.utils.data import Dataset
# from torchtext.data import get_tokenizer
# import requests
# import torch
# import numpy as np
# from PIL import Image
# import pickle
# # from torchvision import transforms
# # from datasets import load_dataset
# # import torch.nn as nn
# import matplotlib.pyplot as plt
# import os
# from tqdm import tqdm

# import warnings
# warnings.filterwarnings('ignore')

# model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer       = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# def show_n_generate(img_path, greedy = True, model = model_raw):
#     image = Image.open(img_path)
#     pixel_values   = image_processor(image, return_tensors ="pt").pixel_values
#     plt.imshow(np.asarray(image))
#     plt.show()

#     if greedy:
#         generated_ids  = model.generate(pixel_values, max_new_tokens = 30)
#     else:
#         generated_ids  = model.generate(
#             pixel_values,
#             do_sample=True,
#             max_new_tokens = 30,
#             top_k=5)
#     generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print(generated_text)

# path = "images/image1.jpg"
# show_n_generate(path, greedy = False)



import streamlit as st
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load models
model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image):
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    # Generate caption
    generated_ids = model_raw.generate(
        pixel_values,
        do_sample=True,
        max_new_tokens=30,
        top_k=5
    )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def main():
    st.title("Image Captioning")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button("Generate Caption"):
            generated_caption = generate_caption(image)
            st.success("Generated Caption: {}".format(generated_caption))

if __name__ == "__main__":
    main()
