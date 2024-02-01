from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

class ImageCaptionGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Caption Generator")

       
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

       
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

     
        self.caption_label = tk.Label(root, text="", wraplength=400, justify=tk.LEFT)
        self.caption_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

        if file_path:
         
            image = Image.open(file_path)
            image = image.resize((500, 500))  
            photo = ImageTk.PhotoImage(image)

     
            self.image_label.config(image=photo)
            self.image_label.image = photo

      
            caption = predict_step([file_path])
            real_caption = ' '.join(caption)

            
            self.caption_label.config(text="Generated Caption:\n" + real_caption)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptionGeneratorApp(root)
    root.mainloop()
