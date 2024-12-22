# Stable Diffusion API Image Generation

## Project Overview
This project demonstrates how to use the Stable Diffusion API to generate images from text prompts within a Jupyter Notebook. The goal was to explore the capabilities of text-to-image generation, providing hands-on experience with Stability AI's API.

## How It Works
- **API Integration**: The notebook sends a POST request to the Stability AI endpoint with a text prompt.
- **Image Generation**: The API generates an image based on the prompt and returns it in `.webp` format.
- **Conversion**: Since `.webp` files are not natively supported by Jupyter Notebooks, the code converts the image to `.png` for display.
- **Display**: The generated image is displayed directly in the notebook for easy visualization.

## Key Features
- Seamless integration with Stability AI’s API
- Simple, configurable prompt-based image generation
- Automatic conversion from `.webp` to `.png` for visualization

## Setup Instructions
```bash
# Clone this repository
git clone https://github.com/yourusername/stable-diffusion-api-project.git

# Install the required dependencies
pip install requests pillow
```
```python
# Add your Stability AI API key to the code
API_KEY = "your_api_key_here"
```

Run the Jupyter Notebook to generate and display images.

## Example Prompt
```python
PROMPT = (
    "Create an image of a dog in front of a Christmas tree that has lights turned on and lots of decorations. "
    "The decorations on the tree are all different animals. The tree should be at least 10 times the size of the dog."
)
```

## Notes
- The API offers **25 free credits** upon registration. Each image generation costs 3 credits.
- Advanced models, such as Ultra, may yield more accurate results but at a higher cost.

## Project Status
This is an initial exploration project. Future enhancements may include:
- Batch image generation
- Testing advanced models
- Implementing more complex prompts and workflows

## Contributing
Feel free to submit issues or pull requests if you’d like to contribute or improve the project.

