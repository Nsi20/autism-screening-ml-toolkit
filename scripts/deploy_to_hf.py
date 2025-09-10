from huggingface_hub import HfApi, create_repo
import os
import sys
from pathlib import Path
import shutil
import subprocess

def ensure_correct_dependencies():
    """Install correct versions of dependencies"""
    requirements = """
numpy==1.24.3
scikit-learn==1.3.1
pandas==2.1.1
gradio==3.50.2
joblib==1.3.2
nbformat
nbconvert
    """.strip()
    
    # Write temporary requirements file
    with open("temp_requirements.txt", "w") as f:
        f.write(requirements)
    
    try:
        subprocess.run(["uv", "pip", "install", "-r", "temp_requirements.txt"], check=True)
    finally:
        os.unlink("temp_requirements.txt")

def deploy_to_huggingface():
    # First ensure correct dependencies
    ensure_correct_dependencies()
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in environment")
        sys.exit(1)

    username = os.getenv("HF_USERNAME")
    if not username:
        print("Error: HF_USERNAME not found in environment")
        sys.exit(1)

    # Project paths
    root_dir = Path(__file__).parent.parent
    model_path = root_dir / "models" / "calibrated_lr.pkl"
    
    # Force model retraining with correct versions
    print("Retraining model with correct package versions...")
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        
        notebook_path = root_dir / "notebooks" / "02_features.ipynb"
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': notebook_path.parent}})
        
        if not model_path.exists():
            print("Error: Model training failed")
            sys.exit(1)
    except Exception as e:
        print(f"Error training model: {e}")
        sys.exit(1)

    repo_id = f"{username}/autism-screening"
    api = HfApi(token=token)
    
    try:
        # Create deployment directory
        deploy_dir = root_dir / "deploy_temp"
        deploy_dir.mkdir(exist_ok=True)
        
        # Copy files to deployment directory
        for item in root_dir.iterdir():
            if item.name not in ['.git', '.venv', '__pycache__', 'deploy_temp']:
                if item.is_dir():
                    shutil.copytree(item, deploy_dir / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, deploy_dir / item.name)

        # Write requirements.txt with exact versions
        with open(deploy_dir / "requirements.txt", "w") as f:
            f.write("""
numpy==1.24.3
scikit-learn==1.3.1
pandas==2.1.1
gradio==3.50.2
joblib==1.3.2
            """.strip())

        # Create runtime.txt
        with open(deploy_dir / "runtime.txt", "w") as f:
            f.write("python-3.10")

        # Upload to Hugging Face
        print(f"Uploading to {repo_id}...")
        api.upload_folder(
            folder_path=str(deploy_dir),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[
                ".*",
                "__pycache__",
                "*.pyc",
                ".git*",
                ".venv/*",
                "data/*",
            ]
        )
        print(f"Successfully deployed to https://huggingface.co/spaces/{repo_id}")
    
    except Exception as e:
        print(f"Error during deployment: {e}")
        sys.exit(1)
    finally:
        if (deploy_dir := root_dir / "deploy_temp").exists():
            shutil.rmtree(deploy_dir)

if __name__ == "__main__":
    deploy_to_huggingface()