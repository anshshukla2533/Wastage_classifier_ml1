# ♻️ Waste Classification System

An AI-powered waste classification web application built using TensorFlow, MobileNetV2, and Streamlit.

The model classifies waste images into the following categories:

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

## Features

* Upload waste images
* Real-time image classification
* Confidence score display
* Responsive Streamlit interface
* Lightweight MobileNetV2 transfer learning model
* Deployed on Streamlit Cloud

## Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* Streamlit
* NumPy
* Pillow

## Dataset

This project uses the TrashNet dataset containing over 2500 waste images across 6 categories.

Classes:

* Cardboard
* Glass
* Metal
* Paper
* Plastic
* Trash

## Model Architecture

* MobileNetV2 (Pretrained on ImageNet)
* Global Average Pooling
* Dense Layer (128 Neurons)
* Softmax Output Layer

## Training Results

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | 85%   |
| Validation Accuracy | 73%   |

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/waste-classification-system.git
cd waste-classification-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
waste-classification-system/
│
├── app.py
├── waste_classifier.keras
├── requirements.txt
├── README.md
└── assets/
```

## Future Improvements

* YOLOv8-based object detection
* Organic waste classification
* E-waste detection
* Smart waste segregation recommendations
* Recycling guidance system

## Author

Ansh Shukla
