# Deploy-Classify-Food101-by-ViT
<h2><center>Build VisionTransformer to classify 50% of Tiny Food101 </center> </h2>
<href> <center>https://arxiv.org/abs/2010.11929</center> </href> <br>
<p1>Framework & Tools: Pytorch, matplotlib, wandb, Flask, Gradio, ... </p1> <br>
Download dataset <br>
```
kaggle datasets download -d msarmi9/food101tiny 
``` <br>
Results: Compare to this paper with not lr warm-up, lr decay <br>

To run: with all hyperparameter are the same as the base version of model <br>

Training 1 epochs around 17 minutes with CPU: AMD Ryzen 5600H, GPU: Laptop RTX 3050 4gb VRAM <br>

```
python main.py
```

<br>

Train/Test loss and accuracy <br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/7727df2f-97cb-48d6-8c88-0f23cace09db" />
</div>

Predict a new image <br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/6ab1080d-25f9-4a00-81ce-46d47160671b" />
</div>

Use pretrained model <br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/44c8b1f3-f963-44e1-a02a-4638513e011f" />
</div> <br>

Log loss and accuracy with wandb <br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/5c410ff8-0aa5-4380-a372-1f4dad65ba58" />
</div>


Use Flask to deploy <br>

```
python app.py
```
<div align="center">
  <img src="https://github.com/user-attachments/assets/0f909b48-5753-45b9-8a5f-26047bd7c605" />
</div>


Use Gradio to deploy Local
```
cd demos/food101-tiny
python app.py
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/008ddcaf-09a4-4f05-97b9-5ef0178a0578" />
</div>







