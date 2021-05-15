from pydantic import BaseSettings

class IndexSettings(BaseSettings):
    name: str = "beir/test"
    data_folder: str = "/home"