import os

from src.utils.colab import is_environment_colab_notebook, access_colab_secret

def get_secret(secret_name: str) -> str:
    if is_environment_colab_notebook():
        return access_colab_secret(secret_name)
    return os.environ.get(secret_name)