from pathlib import Path
import transformers

# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

MAX_LEN = 32
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 8
EPOCHS = 1

# Use local folder
BASE_MODEL_PATH = BASE_DIR / "input" / "bert_base_uncased"

# Dataset 
TRAINING_FILE = BASE_DIR / "input" / "ner_dataset.csv"

# Save/load fine-tuned weights to project root
MODEL_PATH = BASE_DIR / "model.bin"

TOKENIZER = transformers.BertTokenizer.from_pretrained(
    str(BASE_MODEL_PATH),
    do_lower_case=True,
    local_files_only=True
)
