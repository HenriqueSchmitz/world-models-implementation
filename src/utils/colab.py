import os

def is_environment_colab_notebook():
    return "COLAB_RELEASE_TAG" in os.environ

def access_colab_secret(secret_name: str) -> str:
    if not is_environment_colab_notebook():
        raise OSError("Trying to access a secret from colab notebook outside of colab environment")
    from google.colab import userdata
    return userdata.get(secret_name)