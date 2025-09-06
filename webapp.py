import torch
import streamlit as st
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import timm

NUM_CLASSES = 7
IMAGE_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Tea algal leaf spot", "Brown Blight", "Gray Blight", "Helopolis", "Red spider",
    "Green mirid bug", "Healthy Leaf"]

# step 1 | loading the model
@st.cache_resource
def load_model():
    model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("deit_best.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Step 2 | creating an instance of the model
model = load_model()


# Step 3 | transforms
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=imagenet_mean,std=imagenet_std),
                                 transforms.Resize((IMAGE_SIZE,IMAGE_SIZE))])

# Step 3 | user interface
st.title("Tea Leaf Disease Detection WebApp")
st.write("Upload an image of a tea leaf and get the disease prediction with probabilities.")

# Step 3.1 | uploading file
uploaded_file = st.file_uploader("Import and image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # showing the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # preprocessing
    img_tensor = transforms(image).unsqueeze(0)

    # prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        prob, pred_class = torch.max(probs, 1)

    # results
    predicted_label = CLASS_NAMES[pred_class.item()]
    confidence = prob.item() * 100

    st.success(f"✅ Predicted Class: **{predicted_label}** ({confidence:.2f}%)")
