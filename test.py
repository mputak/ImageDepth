from PIL import ImageTk, Image
import tkinter as tk
from tkinter import filedialog, PhotoImage
import customtkinter as ctk
import torch
from diffusers import StableDiffusionDepth2ImgPipeline

class Generator:
    def __init__(self, prompt, filename):
        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_attention_slicing() 

        self.filename = filename
        self.init_image = Image.open(self.filename)
        self.init_image.thumbnail((448, 512))

        self.prompt = prompt
        n_propmt = "bad, deformed, ugly, bad anotomy"

        self.strength = 0.7

        self.image = self.pipe(prompt=prompt, image=self.init_image, negative_prompt=n_propmt, strength=self.strength).images[0]

class App(ctk.CTk):
    def __init__(self):
        ctk.CTk.__init__(self)
        self.title("ImageDepthinator 5000")
        self.geometry("600x700")

        self.image_label = tk.Label(self, borderwidth=0)
        self.image_label.pack()

        self.entry = ctk.CTkEntry(self, placeholder_text="Type prompt here.")
        self.entry.pack()

        self.browse_button = ctk.CTkButton(self, text="Browse", command=self.browse, border_width=1)
        self.browse_button.pack()

        self.generate_button = ctk.CTkButton(self, text="Generate", command=self.generate, border_width=1)
        self.generate_button.pack()

        self.save_button = ctk.CTkButton(self, text="Save image!", command=self.save, border_width=1, fg_color="green")


    def browse(self):
        self.filename = filedialog.askopenfilename()
        print(self.filename)

    def generate(self):
        prompt = self.entry.get()
        self.generator = Generator(prompt, self.filename)

        self.image = ImageTk.PhotoImage(self.generator.image)
        self.image_label.config(image=self.image)
        self.image_label.image = self.image
        self.save_button.pack()

    def save(self):
        self.generator.image.save("generated_image.png")


app = App()
app.mainloop()