import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("waste_classifier.keras")

classes = [
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Trash"
]

st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️"
)

st.title("♻️ Waste Classification System")
st.write("Upload a waste image and the model will classify it.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # Convert all images to RGB
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    # Preprocess image
    img = image.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0]

    predicted_class = np.argmax(prediction)

    confidence = prediction[predicted_class] * 100

    st.subheader("Class Probabilities")

    for i, score in enumerate(prediction):
        st.write(
            f"{classes[i]} : {score*100:.2f}%"
        )

    st.divider()

    # Reject uncertain predictions
    if confidence < 70:
        st.warning(
            f"⚠️ Model is uncertain.\n\nBest Guess: {classes[predicted_class]} ({confidence:.2f}%)"
        )
    else:
        st.success(
            f"✅ Prediction: {classes[predicted_class]}"
        )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )