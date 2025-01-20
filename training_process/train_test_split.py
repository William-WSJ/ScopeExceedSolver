import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.resolve()
sys.path.append(str(project_root))

import json
import random

from global_variable import TEST, TRAIN_DATASET, TEST_DATASET

def read_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def train_test_split(dataset, train_set=TRAIN_DATASET, test_set=TEST_DATASET, ratio=0.8):
    random.shuffle(dataset)
    split_index = int(len(dataset) * ratio)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    
    with open(train_set, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    
    with open(test_set, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    
    return train_data, test_data

def main():
    dataset = read_dataset(TEST)
    train_test_split(dataset)

if __name__=="__main__":
    main()