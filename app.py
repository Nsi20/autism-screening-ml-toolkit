import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# Model loading with better error handling
try:
    MODEL_PATH = os.path.join('models', 'calibrated_lr.pkl')
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found. Please run the training notebook first.")

# Define the Gradio interface
def predict_autism(features):
    # Convert the features into a DataFrame
    features_df = pd.DataFrame([features], columns=[f'A{i}' for i in range(1, 11)] + ['age', 'gender', 'ethnicity', 'country'])
    
    # Ensure the columns are in the right order
    features_df = features_df[[f'A{i}' for i in range(1, 11)] + ['age', 'gender', 'ethnicity', 'country']]
    
    # Make the prediction
    prediction = model.predict(features_df)
    
    return 'ASD' if prediction[0] == 1 else 'No ASD'

# Improved CSS for the Gradio interface
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
    background-color: #f4f4f9;
    color: #333;
}

.gradio-button {
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
}

.gradio-button:hover {
    background-color: #0056b3;
}

.gradio-input {
    border: 1px solid #ced4da;
    border-radius: 5px;
    padding: 10px;
    font-size: 16px;
}

.gradio-label {
    font-weight: bold;
}
"""

# Launch the interface
gr.Interface(
    fn=predict_autism,
    inputs=[gr.inputs.Slider(0, 10, label=f'A{i}') for i in range(1, 11)] + 
           [gr.inputs.Slider(2.7, 89, label='Age'), 
            gr.inputs.Radio(['Male', 'Female'], label='Gender'), 
            gr.inputs.Dropdown(['Ethnicity1', 'Ethnicity2', 'Ethnicity3'], label='Ethnicity'), 
            gr.inputs.Textbox(label='Country')],
    outputs=gr.outputs.Label(num_top_classes=2, label='Prediction'),
    title="Autism Spectrum Disorder (ASD) Prediction",
    description="Enter the responses to the 10 screening questions (A1-A10), age, gender, ethnicity, and country. The model will predict the likelihood of Autism Spectrum Disorder.",
    theme="default",
    css=css
).launch()