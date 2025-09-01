# src/app.py
import os, joblib, pandas as pd, numpy as np
import gradio as gr
from constants import ASD_QUESTIONS

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'calibrated_lr.pkl')
model = joblib.load(MODEL_PATH)

# Update the labels and input structure
labels = [f'A{i}_Score' for i in range(1,11)] + \
         ['age', 'gender', 'jaundice', 'austim', 'ethnicity', 'contry_of_res', 'used_app_before', 'relation']

def predict(*vals):
    # Convert inputs to correct format
    input_dict = dict(zip(labels, vals))
    input_dict['ethnicity'] = input_dict.get('ethnicity', 'Others')
    
    X = pd.DataFrame([input_dict])
    
    try:
        prob = float(model.predict_proba(X)[0,1])
        risk_level = "High risk" if prob >= 0.7 else "Low risk"
        
        # Format response as a proper dictionary
        response = {
            "risk_assessment": {
                "probability": f"{prob:.1%}",
                "risk_level": risk_level
            },
            "disclaimer": "This is a screening tool only. Please consult healthcare professionals for diagnosis."
        }
        return response
        
    except Exception as e:
        return {"error": str(e)}

# Custom CSS for better appearance
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
    max-width: 900px !important;
    margin: auto;
}
.gr-form {
    background-color: #f7f7f7;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.gr-title {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 20px;
}
"""

# Updated description
description = """
### Autism Spectrum Disorder (ASD) Screening Tool
This tool helps assess the risk of ASD based on behavioral and demographic factors.

#### Instructions:
1. Answer the 10 screening questions (A1-A10) with 0 (No) or 1 (Yes)
2. Fill in demographic information
3. Click submit to get the assessment

**Note:** This is a screening tool and not a diagnostic instrument. Please consult healthcare professionals for proper diagnosis.
"""

# ---------- Gradio UI ----------
demo = gr.Interface(
    fn=predict,
    inputs=[
        *[gr.Radio([0, 1], 
                   label=f"Q{i}: {ASD_QUESTIONS[f'A{i}_Score']}", 
                   info="Select 0 for No, 1 for Yes") 
          for i in range(1, 11)],
        gr.Number(label="Age", info="Enter age in years"),
        gr.Radio(['m', 'f'], label="Gender", info="Select gender"),
        gr.Radio(['yes', 'no'], label="Jaundice History", info="History of jaundice"),
        gr.Radio(['yes', 'no'], label="Family History of Autism", info="Select yes if there's family history"),
        gr.Dropdown(
            choices=['White-European', 'Asian', 'Middle Eastern', 'South Asian', 'Hispanic', 'Black', 'Others'],
            label="Ethnicity",
            info="Select ethnicity"
        ),
        gr.Dropdown(choices=['UK', 'US', 'Others'], label="Country", info="Select country"),
        gr.Radio(['yes', 'no'], label="Used App Before", info="Previous usage of the app"),
        gr.Dropdown(choices=['Parent', 'Self', 'Health care professional', 'Relative', 'Others'], 
                   label="Relation", info="Your relation to the subject")
    ],
    outputs=gr.JSON(label="Screening Results"),
    title="ASD Screening Assessment",
    description=description,
    css=css,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray"
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=8080,
        share=True
    )