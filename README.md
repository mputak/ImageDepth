# ImageDepth
This main usage of this project is to use state-of-the-art stable diffusion depth model to redesign original input image per user-defined prompt.

## Prerequisites

Run the following commands in order of appearance:
```python
pip install -r requirements.txt
pip install -U git+https://github.com/huggingface/transformers.git
pip install diffusers transformers accelerate scipy safetensors
```

## How to use

1. Run the program
2. Choose an image you wish to transform
3. Describe what specific changes would you like to make to the image in the text box
4. Press **Generate!** button
5. Save the image if you like the outcome, and if not repeat from **step 3**

## Example

Original image:\
![Lenna](./images/Lenna_(test_image).png)

Prompt: **sunglasses, in space**

Output image:\
![Cool Lenna](./images/Cool_Lenna_(test_image).png)


For any questions do not hesitate to contact me. Have fun!
