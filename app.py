# app.py
import gradio as gr, joblib, pandas as pd, numpy as np, os

# ---------- model ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "calibrated_lr.pkl")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise SystemExit("Model not found – be sure to upload calibrated_lr.pkl into the models/ folder")

# ---------- helpers ----------
A_COLS   = [f"A{i}_Score" for i in range(1, 11)]
CAT_COLS = ["age", "gender", "jaundice", "austim", "contry_of_res", "used_app_before", "relation"]

def predict(*vals):
    X = pd.DataFrame([vals], columns=A_COLS + CAT_COLS)
    prob = float(model.predict_proba(X)[0, 1])
    label = "High risk – consider referral" if prob >= 0.7 else "Low risk"
    return f"{label} (probability {prob:.1%})"

# ---------- Gradio UI ----------
css = """
.gradio-container{font-family:Arial,sans-serif;background:#f4f4f9;color:#333}
"""
iface = gr.Interface(
    fn=predict,
    inputs=[gr.Radio([0, 1], label=f"Question {i}") for i in range(1, 11)] +
           [gr.Number(label="Age", value=25),
            gr.Radio(["m", "f"], label="Gender"),
            gr.Radio(["yes", "no"], label="Jaundice at birth"),
            gr.Radio(["yes", "no"], label="Family member with ASD"),
            gr.Textbox(label="Country", placeholder="India"),
            gr.Radio(["yes", "no"], label="Used app before"),
            gr.Textbox(label="Who filled form", placeholder="Self / Parent / Doctor")],
    outputs=gr.Label(label="Result"),
    title="🧩 Autism Screening Tool",
    description="Answer 10 quick questions and get an instant risk indicator.  \n**Not a diagnosis** – always consult a clinician.",
    css=css,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)