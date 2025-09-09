from huggingface_hub import HfApi, create_repo
import os

def deploy_to_huggingface():
    # Initialize Hugging Face API
    api = HfApi()
    
    # Create Space if it doesn't exist
    try:
        create_repo(
            repo_id="autism-screening",
            repo_type="space",
            space_sdk="gradio",
            private=False
        )
    except Exception as e:
        print(f"Space might already exist: {e}")

    # Upload files in chunks
    api.upload_folder(
        folder_path=".",
        repo_id="autism-screening",
        repo_type="space",
        ignore_patterns=[
            ".*",
            "__pycache__",
            "*.pyc",
            ".git*",
            ".venv/*",
            "data/*",
            "models/*.pkl"
        ]
    )

if __name__ == "__main__":
    deploy_to_huggingface()