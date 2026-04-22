from pathlib import Path

from app.config import settings


if __name__ == "__main__":
    print("Project root:", settings.project_root)
    print("Data dir exists:", Path(settings.data_dir).exists())
    print("Chroma dir exists:", Path(settings.chroma_dir).exists())
    print("Config loaded: OK")

