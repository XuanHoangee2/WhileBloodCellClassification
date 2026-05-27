import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format = '[%(asctime)s]: %(message)s:')
project_name = "WhiteBloodCellClassification"

list_of_file = [
    "data/RawData/.gitkeep",
    "data/SourceData/.gitkeep",
    f"{project_name}/models/PixelEncoder.py",
    f"{project_name}/models/PixelDecoder.py",
    f"{project_name}/models/TransformerDecoder.py",
    f"{project_name}/models/spatial_cooccurrence.py",
    f"{project_name}/models/segmentationHead.py",
    f"{project_name}/models/classifier.py",
    f"{project_name}/models/losses.py",
    f"{project_name}/models/blocks.py",
    f"{project_name}/models/__init__.py",
    f"{project_name}/DomainAdaptation/__init__.py",
    f"{project_name}/DomainAdaptation/DA_module.py",
    f"{project_name}/DomainAdaptation/TA_module.py",
    f"{project_name}/utils.py",
    f"{project_name}/logger.py",
    f"{project_name}/__init__.py",
    f"{project_name}/dataset.py",
    f"{project_name}/DA_training.py",
    f"{project_name}/TA_training.py",
    f"{project_name}/config/config.yaml",
    f"{project_name}/config/config_task.yaml",
    f"{project_name}/config/config_loader.py",
    "research/__init__.py",
    "research/trials.ipynb",
    "requirements.txt",
    "train.py",
    "process_data.py",
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