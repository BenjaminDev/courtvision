from pydantic_settings import BaseSettings
from pathlib import Path

class CourtVisionSettings(BaseSettings):
    datasets_path: Path
    ball_models_dir: Path
    ball_model_name: str
    ball_checkpoints_dir: Path
    ball_checkpoints_weights_only: Path
    
    wb_project: str
    wb_save_dir: str

    class Config:
        extra='ignore'
        env_prefix = "COURTVISION_"
        env_file = ".env"
        env_file_encoding = "utf-8"
