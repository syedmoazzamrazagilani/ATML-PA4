import subprocess, sys
from pathlib import Path
import torch

assert torch.cuda.is_available(), "GPU not enabled!"
print("Using GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)

data_repo = Path("atml-pa4-datasets")
if not data_repo.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/AbDu11aHHH/atml-pa4-datasets.git"],
        check=True,
    )

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(data_repo / "requirements-public.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.1", "huggingface_hub==0.36.2", "peft", "accelerate", "scipy"], check=True)

subprocess.run([sys.executable, str(data_repo / "scripts/verify_checksums.py")], check=True)
subprocess.run([sys.executable, str(data_repo / "scripts/validate_assets.py")], check=True)

subprocess.run([sys.executable, str(data_repo / "scripts/setup_data.py"), "--all"], check=True)
subprocess.run([sys.executable, str(data_repo / "scripts/check_setup.py")], check=True)
