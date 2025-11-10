"""
Main script for Bangla BERT Hate Speech Detection
Runs K-fold cross-validation training for binary classification
"""

from transformers import AutoTokenizer
import torch
import time  # <--- ADD THIS
from config import parse_arguments, print_config
from data import load_and_preprocess_data, prepare_kfold_splits
from train import run_kfold_training
from utils import set_seed


def main():
    # Parse arguments
    config = parse_arguments()
    print_config(config)

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Record start time
    start_time = time.time()  # <--- ADD THIS

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    # Load and preprocess data
    comments, labels = load_and_preprocess_data(config.dataset_path)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Run K-fold training
    run_kfold_training(config, comments, labels, tokenizer, device, start_time)  # <--- UPDATE THIS


if __name__ == "__main__":
    main()
