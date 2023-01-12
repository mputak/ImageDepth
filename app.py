import tkinter as tk
from tkinter import filedialog, PhotoImage
from PIL import ImageTk, Image
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
        self.n_propmt = "bad, deformed, ugly, bad anotomy"
        self.strength = 0.7

        self.image = self.pipe(prompt=prompt, image=self.init_image,
            negative_prompt=self.n_propmt, strength=self.strength).images[0]


class App(ctk.CTk):
    def __init__(self):
        ctk.CTk.__init__(self)
        self.title("ImageDepthinator 5000")
        self.geometry("500x700")
        # Label for image
        self.image_label = tk.Label(self, borderwidth=0)
        self.image_label.place(x=250, y=0, anchor="n")
        # Prompt entry
        self.entry = ctk.CTkEntry(self, placeholder_text="Type prompt here.", width=200)
        # Browse button
        self.browse_button = ctk.CTkButton(self, text="Browse", command=self.browse, border_width=1, width=120)
        self.browse_button.place(x=250, y=530, anchor="center")
        # Generate button
        self.generate_button = ctk.CTkButton(self, text="Generate", command=self.generate, border_width=2, width=160, fg_color="purple")
        # Save button
        self.save_button = ctk.CTkButton(self, text="Save image!", command=self.save, border_width=1, fg_color="green")

    def browse(self):
        '''
        Definition for browsing files in directories.
        Input: None

        Adds: 
        - Chosen image preview,
        - Prompt field,
        - Generate button.
        Return: None.
        '''
        self.filename = filedialog.askopenfilename()
        print(self.filename)
        # Image preview
        pil_img = Image.open(self.filename)
        pil_img.thumbnail((448, 512))
        tk_image = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image
        # Entry and Generate button placement
        self.entry.place(x=250, y=560, anchor="center")
        self.generate_button.place(x=250, y=590, anchor="center")
        
    def generate(self):
        prompt = self.entry.get()
        self.generator = Generator(prompt, self.filename)

        self.image = ImageTk.PhotoImage(self.generator.image)
        self.image_label.config(image=self.image)
        self.image_label.image = self.image
        self.save_button.place(x=250, y=620, anchor="center")

    def save(self):
        self.generator.image.save("generated_image.png")
        saved_label = ctk.CTkLabel(self, text="Image saved!")
        saved_label.place(x=250, y=650, anchor="center")
        self.after(3000, saved_label.destroy)

app = App()
app.mainloop()
