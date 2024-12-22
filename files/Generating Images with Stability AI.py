#!/usr/bin/env python
# coding: utf-8
Author: Cesar Jaidar
# # Generating Images with Stability AI. 

# ## This notebook demonstrates how to use the Stable Diffusion API to generate images based on text prompts.

# In[20]:


import requests
from IPython.display import Image, display
from PIL import Image as PILImage


# ## The code is organized to separate configuration (API key, endpoint, outputfile and prompt) from the request and display logic.

# In[24]:


# Define configuration
API_URL = "https://api.stability.ai/v2beta/stable-image/generate/core" #The core model is their cheaper model
API_KEY = "add your key here"
OUTPUT_FILE = "./dog_christmas_tree.webp"

# We utilized the same prompt that we used in the previoous exercise to see how similar or different this model
# is compater to OpenAI and DeepAI
PROMPT = (
    "Create an image of a dog in front of a Christmas tree that has lights turned on and lots of decorations. The decorations on the tree are all different animals. The tree should be at least 10 times the size of the dog.")


# ## API Request - We send a POST request to the Stability AI endpoint with the desired prompt and configurations.

# In[22]:


# Request payload and headers
payload = {
    "prompt": PROMPT,
    "output_format": "webp"
}

headers = {
    "authorization": f"Bearer {API_KEY}",
    "accept": "image/*"
}


# ## The generated image is saved in .webp based on the API documentation. Since .webp is not directly supported by IPython display, the image is converted to .png for visualization.

# In[23]:


# Image generation request
response = requests.post(API_URL, headers=headers, files={"none": ''}, data=payload)

# Save and display image
if response.status_code == 200:
    with open(OUTPUT_FILE, 'wb') as file:
        file.write(response.content)
    
    # Convert webp to png for display
    def convert_and_display_webp(webp_file, output_file="./converted_image.png"):
        image = PILImage.open(webp_file)
        image.save(output_file, "PNG")
        display(Image(output_file))
    
    convert_and_display_webp(OUTPUT_FILE)
else:
    raise Exception(response.json())


# ## Conclusion
The Stable Diffusion API was easy to work with and integrating it into the project was straightforward. Since the API has a pricing structure, I could only run a few tests before running out of the 25 free credits—each image cost 3 credits.

One small hiccup was dealing with `.webp` files. Jupyter Notebooks doesn’t display them natively, so I had to convert them to .png, which added an extra step.

Compared to the images generated by OpenAI and DeepAI in the previous exercise, this one seems more realistic to me (even though there are inconsistencies in the image, like the back-right leg of the dog being miss-aligned). The model did miss part of the prompt about animal decorations on the tree and used more tranditional decorations, which might be because I used the “core” model. This is the cheaper option, and Stability AI offers more advanced models like Ultra that might handle prompts more accurately, but at a higher cost.