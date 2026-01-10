import transformers
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 4
BASE_MODEL_PATH = "bert-base-uncased"
MODEL_PATH = 'model.bin'
TRAINING_FILE = os.path.join(BASE_DIR, "input", "ner_dataset.csv")
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case = True
)
