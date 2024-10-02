from typing import Tuple, Dict
from timeit import default_timer as timer
import torch
import os
import json
from PIL import Image
import model
import io
import matplotlib.pyplot as plt
from PIL import Image
from model import create_vit_best_model
import gradio as gr 

# food101_image_class = json.load(open('food101_image_class.json'))
food101_image_class = ["apple_pie", "bibimbap", "cannoli", 
                       "edamame", "falafel", "french_toast", 
                       "ice_cream", "ramen", "sushi", "tiramisu"]
vit_model, transform = create_vit_best_model(num_classes=10, seed=42)
vit_model.load_state_dict(torch.load('vit.pt', weights_only=True))
vit_model.to(device='cpu')

def predict_image(img) -> Tuple[Dict, float]:
    start_time = timer()
    img = transform(img).unsqueeze(0)
    vit_model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(vit_model(img), dim=1)
    pred_class_probs = {food101_image_class[i]: float(pred_probs[0][i]) for i in range(len(food101_image_class))}
    pred_time = round(timer() - start_time, 5)

    return pred_class_probs, pred_time


title = "Image Classification of TinyFood101"
description = "<center> Vision Transformer to extract features from images"
artical = "<center>Created by PGP for Internship"
example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict_image, 
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=10, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=artical
                    )

# shutdown_button = gr.Button("Shutdown")
# shutdown_button.click(lambda: os._exit(0))
demo.launch()
