from PIL import ImageTk, Image
import tkinter as tk
import customtkinter as ctk
import torch
from diffusers import StableDiffusionDepth2ImgPipeline


root = ctk.CTk()
root.geometry("600x400")
root.title("ImageDepthinator 5000")
root.eval('tk::PlaceWindow . center')
ctk.set_appearance_mode("system")
filename = None


class Generator:
   def __init__(self, prompt, filename):

        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
        ).to("cuda")
        self.pipe.enable_attention_slicing() 

        self.filename = filename
        self.init_image = Image.open(self.filename)
        self.init_image = self.init_image.resize((480, 580))

        self.prompt = prompt
        n_propmt = "bad, deformed, ugly, bad anotomy"

        self.strength = 0.7

        self.image = self.pipe(prompt=prompt, image=self.init_image, negative_prompt=n_propmt, strength=self.strength).images #[0]
        print(type(self.image))
        print(self.image)
        # self.init_image.show()
        # self.image[3].show()



class App:
    def __init__(self, root):

        ctk.set_appearance_mode("system")
        self.root = root
        self.root.geometry("600x400")
        self.root.title("ImageDepthinator 5000")
        self.root.eval('tk::PlaceWindow . center')

        self.button_explore = ctk.CTkButton(self.root, text = "Browse Photos",
            command = self.browseFiles, border_width=1, border_color="white")
        self.button_explore.pack()

        self.label = ctk.CTkLabel(self.root ,height=200, width=200)
        self.label.pack()
        self.prompt = ctk.CTkEntry(self.root, placeholder_text="Add your prompt here.")
        self.prompt.pack()

        self.button_generate = ctk.CTkButton(self.root, text = "Generate!",
            command = self.generate, border_width=1, border_color="white")
        self.button_generate.pack()


    def browseFiles(self):
        self.filename = tk.filedialog.askopenfilename(title = "Select a Photo")
        self.button_explore.pack_forget()


    def generate(self):
        prompt = self.prompt.get()
        generator = Generator(prompt, self.filename) # Fix this...

        for image in generator.image:

            tk_img = ImageTk.PhotoImage(image) 
            self.label.configure(image=tk_img)



root = ctk.CTk()
app = App(root)
root.mainloop()
