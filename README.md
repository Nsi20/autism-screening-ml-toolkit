# Autism Spectrum Disorder Screening ML Toolkit

A machine learning-based screening tool for early detection of Autism Spectrum Disorder (ASD) using behavioral markers and demographic information.

## 🎯 Project Overview

This toolkit provides:
- Interactive web interface for ASD screening
- ML model trained on behavioral and demographic data
- Fair assessment across different demographic groups
- Confidence scores with uncertainty estimates

## 🔧 Technology Stack

### Core Technologies
- Python 3.10
- scikit-learn (ML modeling)
- Gradio (web interface)
- Pandas (data processing)
- Jupyter (analysis notebooks)

### Development Tools
- uv (package management)
- Git (version control)
- VSCode (IDE)

## 📁 Project Structure
```
autism-screening-ml-toolkit/
├── data/               # Dataset files
├── models/            # Trained ML models
├── notebooks/         # Jupyter notebooks
│   ├── 01_eda.ipynb  # Exploratory Data Analysis
│   └── 02_features.ipynb  # Feature Engineering
├── src/              # Source code
│   ├── app.py        # Gradio web interface
│   └── constants.py  # Constants and configurations
└── requirements.txt  # Project dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- uv package manager

### Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd autism-screening-ml-toolkit
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Running the Application
1. Start the web interface:
```bash
uv run python src/app.py
```
2. Open browser at http://127.0.0.1:8080

## 📊 Model Details

### Features Used
- 10 behavioral markers (A1-A10 scores)
- Demographic information (age, gender, ethnicity)
- Family history
- Geographic location

### Performance Metrics
- AUC Score: 0.83
- Balanced across demographic groups
- Uncertainty estimates for small group sizes

## 👥 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push branch (`git push origin feature/name`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citations

If you use this toolkit in your research, please cite:

```bibtex
@software{asd_screening_toolkit,
  title={Autism Screening ML Toolkit},
  year={2023},
  author={[Nsidibe Daniel]},
  url={[repository-url]}
}
`