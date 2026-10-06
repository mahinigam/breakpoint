import yaml
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load settings
with open(ROOT_DIR / "config" / "settings.yaml", "r") as f:
    _settings = yaml.safe_load(f)

SEED = _settings["seed"]
NEGATIVE_WEIGHT = _settings["negative_weight"]
TRAIN_VAL_SPLIT_RATIO = _settings["train_val_split_ratio"]

# Paths
PATHS = _settings["paths"]
for key, val in PATHS.items():
    PATHS[key] = ROOT_DIR / val

# Load economic scenarios
with open(ROOT_DIR / "config" / "economic_scenarios.yaml", "r") as f:
    ECONOMIC_SCENARIOS = yaml.safe_load(f)["scenarios"]
