# WebApp-With-Streamlit-For-Tea-Leaf-Disease-Detection
End-To-End Tea Leaf Disease Detection WebApp With Streamlit


## Overview
This project provides an end-to-end solution for detecting diseases in tea leaves using a pre-trained DeiT (Data-efficient Image Transformers) model. The notebook includes data preparation, model training, evaluation, and deployment as a web application using Streamlit. The dataset used is the TeaLeafBD dataset from [kaggle](https://www.kaggle.com/datasets/bmshahriaalam/tealeafbd-tea-leaf-disease-detection), containing images of tea leaves affected by various diseases.

### Key Features
- Fine-tuned pre-trained DeiT model for multi-class classification (7 disease classes).
- Data splitting, augmentation, and loading with PyTorch.
- Training with class weights, metrics (accuracy, precision, recall, F1-score), and visualizations.
- Web app for uploading images and predicting diseases.
- 

## Installation
To run the notebook or the web app, install the required dependencies:

```bash
pip install torch torchvision timm pytorch-lightning transformers split-folders matplotlib seaborn scikit-learn streamlit
```
## Deploying the Web App
```bash
streamlit run app.py
```

## License
https://creativecommons.org/licenses/by-sa/4.0/


## Acknowledgments
DeiT model from the TIMM library.
TeaLeafBD dataset from Kaggle.
Built with PyTorch, PyTorch Lightning, and Streamlit.
