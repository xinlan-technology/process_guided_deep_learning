# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import pandas as pd
from datetime import datetime
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

# Define save location on Google Drive
GOOGLE_DRIVE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Pretraining'
MODEL_OUTPUT_DIR = os.path.join(GOOGLE_DRIVE_BASE, 'Model Output')
PRETRAINED_MODEL_DIR = os.path.join(MODEL_OUTPUT_DIR, 'Pretrained_Models')

# Pretraining data split ratios
PRETRAIN_SPLIT_RATIOS = [0.8, 0.2, 0.0]    # [train, val, test] for pretraining - no test needed

# Create directories
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)

def save_pretrained_model(trial, model_name, save_dir):
    """Save pretrained model and parameters"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f'pretrained_model_{timestamp}.pth')
    params_path = os.path.join(model_dir, f'pretrained_params_{timestamp}.json')

    torch.save(trial['model_state'], model_path)
    with open(params_path, 'w') as f:
        json.dump(trial['params'], f, indent=4)

    print(f"Saved pretrained {model_name} to {model_dir}")
    return model_path, params_path

def pretrain_all_models():
    """Phase 1: Pretrain all models on 100% simulation data"""
    print("="*70)
    print("STEP 1: PRETRAINING ALL MODELS ON 100% SIMULATION DATA")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Prepare simulation data with 100% data
    processor = SequenceProcessor()
    sim_dataloaders, input_size, output_size = processor.prepare_sequence_data(
        use_simulation=True,  # Use simulation data for pretraining
        data_fraction=1.0,    # Use 100% of simulation data
        split_ratios=PRETRAIN_SPLIT_RATIOS  # Use configured split ratios
    )

    pretrained_models = {}

    for i, (model_name, model_class) in enumerate(MODELS.items(), 1):
        print(f"\n[{i}/{len(MODELS)}] Pretraining {model_name} on 100% simulation data...")

        try:
            best_trial = run_optimization(
                model_class=model_class,
                train_loader=sim_dataloaders['train'],
                valid_loader=sim_dataloaders['val'],
                input_size=input_size,
                output_size=output_size
            )
            
            model_path, params_path = save_pretrained_model(
                best_trial, model_name, PRETRAINED_MODEL_DIR
            )
            
            pretrained_models[model_name] = {
                'model_path': model_path,
                'params_path': params_path,
                'params': best_trial['params'],
                'input_size': input_size,
                'output_size': output_size,
                'validation_loss': best_trial['loss']
            }

            print(f"{model_name} pretraining completed! Validation loss: {best_trial['loss']:.6f}")

        except Exception as e:
            print(f"Error pretraining {model_name}: {e}")
            import traceback
            traceback.print_exc()

    return pretrained_models

def main():
    """Main pretraining pipeline"""
    try:
        print("Starting pretraining pipeline...")
        start_time = datetime.now()
        
        # Pretrain all models
        pretrained_models = pretrain_all_models()
        
        if not pretrained_models:
            print("ERROR: No models were successfully pretrained!")
            return
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("PRETRAINING PHASE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print(f"Summary:")
        print(f"   Successfully pretrained models: {len(pretrained_models)}")
        print(f"   Total time: {duration}")
        print(f"   Models saved to: {PRETRAINED_MODEL_DIR}")
        
        print(f"\nValidation losses:")
        for model_name, info in pretrained_models.items():
            print(f"   {model_name}: {info['validation_loss']:.6f}")
        
    except Exception as e:
        print(f"FATAL ERROR in pretraining pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()