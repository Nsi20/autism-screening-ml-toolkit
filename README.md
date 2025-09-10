---
title: autism-screening-ml-toolkit
app_file: app.py
sdk: gradio
sdk_version: 5.44.1
---
# Autism Spectrum Disorder Screening Tool

Machine learning-based screening tool for early detection of Autism Spectrum Disorder (ASD).

## 🎯 Features
- Interactive web interface for ASD screening
- ML model with 83% AUC score
- Fair assessment across demographics
- Uncertainty estimation for predictions

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/autism-screening-ml-toolkit
cd autism-screening-ml-toolkit

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Run web interface
uv run python src/app.py
```

## 📊 Model Training
1. Open `notebooks/01_eda.ipynb` for data exploration
2. Run `notebooks/02_features.ipynb` to train model

## 🌐 Deployment
View the live demo: [Hugging Face Space](https://huggingface.co/spaces/yourusername/autism-screening)

## 📄 License
MIT License