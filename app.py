import tkinter as tk
from tkinter import filedialog, PhotoImage
from PIL import ImageTk, Image
import customtkinter as ctk
import torch
from diffusers import StableDiffusionDepth2ImgPipeline

# add img parameter
class Generator:
    def __init__(self, prompt, init_image):
        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_attention_slicing() 
        
        self.init_image = init_image
        # self.filename = filename
        # self.init_image = Image.open(self.filename)
        # self.init_image.thumbnail((448, 512))

        self.prompt = prompt
        self.n_propmt = "bad, deformed, ugly, bad anotomy"
        self.strength = 0.7

        self.image = self.pipe(prompt=prompt, image=self.init_image,
            negative_prompt=self.n_propmt, strength=self.strength).images[0]


class App(ctk.CTk):
    def __init__(self):
        ctk.CTk.__init__(self)
        self.title("ImageDepthinator 5000")
        self.geometry("600x700")

        self.image_label = tk.Label(self, borderwidth=0)
        self.image_label.place(x=300, y=0, anchor="n")

        self.entry = ctk.CTkEntry(self, placeholder_text="Type prompt here.", width=200)

        self.browse_button = ctk.CTkButton(self, text="Browse", command=self.browse, border_width=1, width=120)
        self.browse_button.place(x=300, y=530, anchor="center")

        self.generate_button = ctk.CTkButton(self, text="Generate", command=self.generate, border_width=2, width=160, fg_color="purple")

        self.save_button = ctk.CTkButton(self, text="Save image!", command=self.save, border_width=1, fg_color="green")

    
    def rescale_image(self, filename):
        im = Image.open(filename)
        width, height = im.size
        new_width = (((width + 63) // 64) * 64)
        new_height = (((height + 63) // 64) * 64)

        if new_width > new_height:
            if new_width > 512:
                new_width = 512
        else:
            if new_height > 512:
                new_height = 512

        aspect_ratio = width / height

        if aspect_ratio > 1:
            im = im.resize((new_width, int(new_width / aspect_ratio)), Image.LANCZOS)
        else:
            im = im.resize((int(new_height * aspect_ratio), new_height), Image.LANCZOS)

        return im

    def browse(self):
        self.filename = filedialog.askopenfilename()
        print(self.filename)

        # Image preview
        # pil_img = Image.open(self.filename)
        # pil_img.thumbnail((448, 512))
        self.pil_img = self.rescale_image(self.filename)
        tk_image = ImageTk.PhotoImage(self.pil_img)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image

        self.entry.place(x=300, y=560, anchor="center")
        self.generate_button.place(x=300, y=590, anchor="center")
        
    def generate(self):
        prompt = self.entry.get()
        self.generator = Generator(prompt, self.pil_img)

        self.image = ImageTk.PhotoImage(self.generator.image)
        self.image_label.config(image=self.image)
        self.image_label.image = self.image

        self.save_button.place(x=300, y=620, anchor="center")

    def save(self):
        self.generator.image.save("generated_image.png")

        saved_label = ctk.CTkLabel(self, text="Image saved!")
        saved_label.place(x=300, y=650, anchor="center")
        self.after(3000, saved_label.destroy)

app = App()
app.mainloop()
