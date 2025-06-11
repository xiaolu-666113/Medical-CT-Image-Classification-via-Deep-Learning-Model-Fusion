```
# 🧠 COVID Image Classification with Model Fusion

This project implements a COVID-related image classification system using **PyTorch** and the **[timm](https://github.com/huggingface/pytorch-image-models)** library. It leverages **model fusion** to combine multiple vision architectures into a more robust classifier.

## 📁 Project Structure
```

├── config.py      # Hyperparameter and directory settings

├── data_loader.py   # Image loading and preprocessing

├── models.py      # Model selection and FusionModel definition

├── train.py      # Training loop with early stopping

├── test.py       # Model evaluation on validation set

├── covidData/     # Directory containing images (organized in subfolders per class)

└── best_model.pth   # Saved best-performing model weights

```
---

## ⚙️ Dependencies

Make sure Python 3.8+ is installed, then run:

```bash
pip install torch torchvision timm
```

To use a GPU, ensure your PyTorch installation supports CUDA.



------





## **📂 Dataset Format**





Put your image data inside the covidData/ directory like this:

```
covidData/
├── Normal/
│   ├── img1.png
│   └── ...
└── COVID/
    ├── img2.png
    └── ...
```

Each subfolder name will be used as the class label.



------





## **⚙️ Configuration (config.py)**



```
batch_size = 4
num_classes = 2
num_epochs = 10
lr = 1e-4
image_size = 224
data_dir = "covidData"
model_names = ['resnet', 'densenet', 'efficientnet', 'vit', 'convnext', 'regnet', 'swin', 'mobilenetv3', 'beit']
```

Modify this file to adjust hyperparameters and selected models.



------





## **🏋️‍♂️ Training**





To train the fusion model, run:

```
python train.py
```



### **Features:**





- Combines 9 popular vision models using weighted averaging
- Displays per-model fusion weights after each epoch
- Saves best model as best_model.pth
- Supports **early stopping** after 5 epochs with no validation improvement





------





## **🧪 Evaluation**





To evaluate the saved model on the validation set, run:

```
python test.py
```

This will:



- Load best_model.pth
- Report accuracy and average loss on the validation set





------





## **🔁 Fusion Strategy (FusionModel)**





The FusionModel class:



- Forwards input through all base models in parallel
- Applies softmax-normalized, learnable weights to logits from each model
- Combines results via weighted sum
- Prints fusion weight distribution after each epoch





------





## **🧾 Example Output**



```
[Epoch 3] Train Loss: 0.1043, Acc: 0.9844 | Val Loss: 0.2381, Acc: 0.9250
📊 Fusion Weights:
  resnet: 0.1823
  densenet: 0.1117
  efficientnet: 0.1601
  vit: 0.1000
  ...
✅ Best model saved! Val Acc: 0.9250
```



------





## **⚠️ Notes**





- Images are converted from grayscale to 3-channel format for compatibility with pretrained models
- Be careful of overfitting if dataset is small or imbalanced
- Make sure num_classes in config.py matches your dataset





------





## **💬 Contact & Contributions**





Feel free to open an Issue or submit a Pull Request if you’d like to contribute or report a bug.
