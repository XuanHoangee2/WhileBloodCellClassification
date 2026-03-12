import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format = '[%(asctime)s]: %(message)s:')
project_name = "WhiteBloodCellClassification"

list_of_file = [
    "data/.gitkeep",
    f"{project_name}/models/PixelEncoder.py",
    f"{project_name}/models/PixelDecoder.py",
    f"{project_name}/models/TransformerDecoder.py",
    f"{project_name}/models/spatial_cooccurrence.py",
    f"{project_name}/models/classifier.py",
    f"{project_name}/models/losses.py",
    f"{project_name}/models/blocks.py",
    "requirements.txt",
    "train.py",
    "utils",
    "configs/configs.yaml",
    "experiments/checkpoints",
]

for filepath in list_of_file:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for this file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filename}")
    else:
        logging.info(f"{filename} is already created")