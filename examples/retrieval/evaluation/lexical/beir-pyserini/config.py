from pydantic import BaseSettings

class IndexSettings(BaseSettings):
    index_name: str = "beir/test"
    data_folder: str = "/home/datasets/"