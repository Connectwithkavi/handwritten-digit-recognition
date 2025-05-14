# deploy_gradio.py

import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the saved model
model = load_model("mnist_cnn_model.h5")

# Prediction function
def predict_digit(img):
    img = img.convert("L").resize((28, 28))  # convert to grayscale and resize
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img)
    return str(np.argmax(prediction))

# Gradio interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", label="Draw a digit"),
    outputs=gr.Label(num_top_classes=1, label="Predicted Digit"),
    title="Handwritten Digit Recognition"
)

# Launch the interface
interface.launch(share=True)
