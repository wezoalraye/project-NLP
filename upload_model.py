from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="mwael399/sentiment-model", exist_ok=True)
api.upload_folder(
    folder_path="sentiment_model",
    repo_id="mwael399/sentiment-model",
)
print("Done — check https://huggingface.co/mwael399/sentiment-model")