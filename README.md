# 🧠 Object Detection Using Faster R-CNN (ResNet-50 + FPN)

This project is part of an AI internship assignment where I built an object detection model by integrating detection layers on top of a pre-trained CNN backbone. The project uses the Faster R-CNN architecture with a ResNet-50 backbone and Feature Pyramid Network (FPN), trained on a COCO-format dataset.

## 🚀 Project Overview

The goal of this project was to:
- Implement an object detection model using a pre-trained CNN.
- Train and evaluate it on a COCO-format dataset (COCO128 from Roboflow).
- Explore AI-assisted development using ChatGPT for debugging and refinement.

---

## 📁 Project Structure

Object-detection-project/
│
├── data/
│ ├── images/ # Training/validation images
│ └── coco_annotations.json # Annotations in COCO format
│
├── src/
│ ├── dataset.py # Custom dataset loader using COCO
│ ├── model.py # Model architecture setup
│ ├── train.py # Training loop
│ ├── eval.py # Evaluation script
│ └── test.py # Inference and visualization
│
├── requirements.txt # Required libraries
└── README.md # Project documentation (this file)
|__ objectdetection.ipynb

## 🛠️ Technologies Used

- Python 3.10.4
- PyTorch 2.7.0
- TorchVision 0.22.0
- COCO API (`pycocotools`)
- Jupyter Notebook & VS Code
- Matplotlib, PIL, NumPy

---

## 🧠 Model Details

- **Backbone:** ResNet-50 with FPN
- **Detection Head:** Faster R-CNN
- **Framework:** TorchVision's `fasterrcnn_resnet50_fpn`
- **Classes:** 91 (from COCO)

---

## 📊 Dataset

- **Source:** Roboflow
- **Format:** COCO 128 (128 images, 640x640, `.coco`)
- Used `pycocotools` to parse and use annotations.

---

## ⚙️ Installation

1. Clone the repository:
bash
git clone https://github.com/yourusername/object-detection-project.git
cd object-detection-project

2. Set up a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3.Install the dependencies:
pip install -r requirements.txt

4.If needed, install specific versions:
pip install torch==2.7.0 torchvision==0.22.0 pycocotools

🏋️ Training
python src/train.py
You can also run the code in Jupyter Notebook format for visual tracking of loss and intermediate outputs.

📈 Evaluation
python src/eval.py
The model is evaluated using common metrics like mAP, precision, and recall.

📷 Inference Example
python src/test.py
Or visualize predictions with bounding boxes in a Jupyter cell.

🤖 AI Assistance
ChatGPT was used for:

Debugging ImportError and torchvision module issues.

Fixing dataset loading logic in __getitem__.

Suggesting correct structure for the training loop and loss handling.

Visualizing predictions using matplotlib.

Most corrections were guided through prompts and refined by understanding the code logic and running tests.

✅ What Went Right
Successfully trained and evaluated Faster R-CNN with custom COCO dataset.

Learned to handle real-world dataset loading and training edge cases.

Understood how to guide and correct AI-generated code.

❌ What Went Wrong
Faced repeated import/module issues in VS Code.

Spent extra hours debugging due to a faulty virtual environment.

Missed the submission deadline by a few hours due to perfectionism and setup issues.

📄 Experience Report
My detailed experience and learnings from this project are documented in the Experience_Report.docx.

🙌 Acknowledgements
TorchVision Models

COCO Dataset

Roboflow

ChatGPT for code assistance and debugging tips.
