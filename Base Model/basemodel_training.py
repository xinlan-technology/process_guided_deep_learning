# Import libraries
import torch
import os
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np

from data_preprocessing import DataProcessor
from sequence_preparation import SequenceProcessor
from model_architecture import LSTMModel, TransformerModel, CNN_LSTM, AttentionLSTM
from parameters_tuning import run_optimization, set_seed

# Set global seed for reproducibility
set_seed(42)

# Define model types
MODELS = {
    'LSTM': LSTMModel,
    'Transformer': TransformerModel,
    'CNN_LSTM': CNN_LSTM,
    'AttentionLSTM': AttentionLSTM
}

# Define training data proportions to test
DATA_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Define save location on Google Drive
GOOGLE_DRIVE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Base Model'
MODEL_OUTPUT_DIR = os.path.join(GOOGLE_DRIVE_BASE, 'Model Output')
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def save_trial_to_drive(trial, model_name, fraction):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(MODEL_OUTPUT_DIR, f'{model_name}_frac{int(fraction*100)}')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f'best_model_{timestamp}.pth')
    params_path = os.path.join(model_dir, f'best_params_{timestamp}.json')

    torch.save(trial['model_state'], model_path)
    with open(params_path, 'w') as f:
        json.dump(trial['params'], f, indent=4)

    print(f"Saved {model_name} @ {int(fraction*100)}% to {model_dir}")

def train_all_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    for fraction in DATA_FRACTIONS:
        print(f"\n----- Training with {int(fraction*100)}% of data -----")

        processor = SequenceProcessor()
        dataloaders, input_size, output_size, _ = processor.prepare_sequence_data(
            use_simulation=False,
            data_fraction=fraction
        )

        for model_name, model_class in MODELS.items():
            print(f"\nOptimizing {model_name} with {int(fraction*100)}% of data")

            try:
                best_trial = run_optimization(
                    model_class=model_class,
                    train_loader=dataloaders['train'],
                    valid_loader=dataloaders['val'],
                    input_size=input_size,
                    output_size=output_size
                )
                save_trial_to_drive(best_trial, model_name, fraction)

            except Exception as e:
                print(f"Error with {model_name} @ {int(fraction*100)}%: {e}")

if __name__ == "__main__":
    try:
        train_all_models()
        print("\nAll training completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")