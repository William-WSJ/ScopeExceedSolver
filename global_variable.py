from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

TEST = PROJECT_ROOT / 'dataset' / 'test.json'
MY_DATASET = PROJECT_ROOT / 'dataset' / 'scope_exceed_dataset.json'
OPEN_DATASET = PROJECT_ROOT / 'dataset' / 'open_source_dataset.json'

TRAIN_DATASET = PROJECT_ROOT / 'training_process' / 'train_set.json'
TEST_DATASET = PROJECT_ROOT / 'training_process' / 'test_set.json'