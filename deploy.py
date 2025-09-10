from huggingface_hub import HfApi
import os

def deploy():
    # Initialize Hugging Face API
    api = HfApi()
    
    # Get token from environment
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found. Please set it in your environment.")

    # Create the space if it doesn't exist
    try:
        api.create_repo(
            repo_id="autism-screening",
            repo_type="space",
            space_sdk="gradio",
            private=False
        )
    except Exception as e:
        print(f"Space might already exist: {e}")

    # Upload files using large folder API
    api.upload_folder(
        folder_path=".",
        repo_id=f"{os.getenv('USERNAME')}/autism-screening",
        repo_type="space",
        ignore_patterns=[
            ".git*",
            ".env",
            "__pycache__",
            "*.pyc",
            ".venv",
            "data/*",
            "*.pkl"
        ]
    )

if __name__ == "__main__":
    deploy()