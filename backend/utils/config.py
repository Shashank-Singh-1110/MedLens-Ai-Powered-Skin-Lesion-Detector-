from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

CLASSES = {
    "akiec": {"name": "Actinic Keratoses", "severity": "pre-cancerous", "urgency": "soon"},
    "bcc": {"name": "Basal Cell Carcinoma", "severity": "cancerous", "urgency": "urgent"},
    "bkl": {"name": "Benign Keratosis", "severity": "benign", "urgency": "routine"},
    "df": {"name": "Dermatofibroma", "severity": "benign", "urgency": "routine"},
    "mel": {"name": "Melanoma", "severity": "cancerous-dangerous", "urgency": "immediate"},
    "nv": {"name": "Melanocytic Nevi (Moles)", "severity": "benign", "urgency": "routine"},
    "vasc": {"name": "Vascular Lesions", "severity": "benign", "urgency": "routine"},
}

CLASS_LIST = sorted(CLASSES.keys())
NUM_CLASSES = len(CLASS_LIST)

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5
MODEL_NAME = "efficientnet_b0"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"