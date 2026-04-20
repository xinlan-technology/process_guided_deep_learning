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
from parameters_tuning import set_seed

# CHANGE THIS VALUE FOR EACH SCRIPT:
# For single fraction: TARGET_FRACTIONS = 0.2
# For multiple fractions: TARGET_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
TARGET_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

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
FINETUNED_MODEL_DIR = os.path.join(MODEL_OUTPUT_DIR, 'Finetuned_Models')

# Finetuning parameters
FINETUNE_LR_FACTOR = 0.5      # Learning rate multiplier for finetuning
FINETUNE_MAX_EPOCHS = 400     # Maximum epochs for finetuning
FINETUNE_PATIENCE = 150       # Early stopping patience for finetuning

# Data split ratios for finetuning
FINETUNE_SPLIT_RATIOS = [0.7, 0.2, 0.1]    # [train, val, test]

# Create directories
os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)

def find_model_files(model_dir):
    """Find model and parameter files in directory"""
    if not os.path.exists(model_dir):
        return None, None
    
    param_files = [f for f in os.listdir(model_dir) if f.startswith('pretrained_params_') and f.endswith('.json')]
    model_files = [f for f in os.listdir(model_dir) if f.startswith('pretrained_model_') and f.endswith('.pth')]
    
    if not param_files or not model_files:
        return None, None
    
    # Get the most recent files
    param_files.sort(reverse=True)
    model_files.sort(reverse=True)
    
    return os.path.join(model_dir, model_files[0]), os.path.join(model_dir, param_files[0])

def load_pretrained_models():
    """Load all pretrained models from Step 1"""
    print("Loading pretrained models...")
    pretrained_models = {}
    
    for model_name in MODELS.keys():
        model_dir = os.path.join(PRETRAINED_MODEL_DIR, model_name)
        model_path, params_path = find_model_files(model_dir)
        
        if model_path and params_path:
            try:
                with open(params_path, 'r') as f:
                    params = json.load(f)
                
                pretrained_models[model_name] = {
                    'model_path': model_path,
                    'params_path': params_path,
                    'params': params,
                    'input_size': 7,   # Weather features
                    'output_size': 21  # Depth levels
                }
                print(f"   Loaded {model_name}")
            except Exception as e:
                print(f"   Failed to load {model_name}: {e}")
        else:
            print(f"   No pretrained model found for {model_name}")
    
    return pretrained_models

def create_model_with_params(model_name, model_class, params, input_size, output_size, device):
    """Create model with given parameters"""
    if model_name == 'Transformer':
        cfg = params['transformer_config']
        model = model_class(
            input_size=input_size,
            d_model=cfg['d_model'],
            nhead=cfg['nhead'],
            output_size=output_size,
            dropout_rate=params['dropout_rate'],
            num_layers=params['num_layers'],
            dim_feedforward_factor=params['dim_feedforward_factor']
        ).to(device)
    elif model_name == 'CNN_LSTM':
        model = model_class(
            input_size=input_size,
            hidden_size1=params['hidden_size'],
            hidden_size2=params['hidden_size'] // 2,
            hidden_size3=params['hidden_size'] // 4,
            output_size=output_size,
            dropout_rate=params['dropout_rate'],
            num_filters_l1=params['num_filters_l1'],
            num_filters_l2=params['num_filters_l2'],
            num_filters_l3=params['num_filters_l3'],
            kernel_size=params['kernel_size']
        ).to(device)
    elif model_name in ['LSTM', 'AttentionLSTM']:
        model = model_class(
            input_size=input_size,
            hidden_size1=params['hidden_size'],
            hidden_size2=params['hidden_size'] // 2,
            hidden_size3=params['hidden_size'] // 4,
            output_size=output_size,
            dropout_rate=params['dropout_rate']
        ).to(device)
    
    return model

def save_finetuned_model(trial, model_name, fraction):
    """Save finetuned model and parameters"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(FINETUNED_MODEL_DIR, f"{model_name}_frac{int(fraction*100)}")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f'finetuned_model_{timestamp}.pth')
    params_path = os.path.join(model_dir, f'finetuned_params_{timestamp}.json')

    torch.save(trial['model_state'], model_path)
    with open(params_path, 'w') as f:
        json.dump(trial['params'], f, indent=4)

    print(f"Saved finetuned {model_name} to {model_dir}")
    return model_path, params_path

def finetune_model(model, train_loader, val_loader, learning_rate, device, max_epochs=400, patience=100):
    """Finetune model with fixed architecture and smaller learning rate"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    print(f"Starting finetuning with lr={learning_rate:.6f}, max_epochs={max_epochs}")
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            X, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                X, y = batch[0].to(device), batch[1].to(device)
                loss = criterion(model(X), y)
                val_loss += loss.item()
                count += 1
        
        val_loss = val_loss / max(1, count)
        train_loss = train_loss / len(train_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f} [BEST]")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return {
        'model_state': best_state,
        'loss': best_val_loss,
        'status': 'OK'
    }

def finetune_for_fraction(pretrained_models, target_fraction):
    """Finetune all models for a specific data fraction"""
    print(f"\nFINETUNING WITH {int(target_fraction*100)}% OF OBSERVATION DATA")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare observation data with target fraction
    processor = SequenceProcessor()
    obs_dataloaders, input_size, output_size, _ = processor.prepare_sequence_data(
        use_simulation=False,  # Use real observation data
        data_fraction=target_fraction,  # Use specified fraction
        split_ratios=FINETUNE_SPLIT_RATIOS  # Use configured split ratios
    )

    finetuned_results = {}

    for i, (model_name, pretrained_info) in enumerate(pretrained_models.items(), 1):
        print(f"\n[{i}/{len(pretrained_models)}] Finetuning {model_name} with {int(target_fraction*100)}% data...")

        try:
            model_class = MODELS[model_name]
            params = pretrained_info['params']
            
            # Create model with SAME architecture as pretrained
            model = create_model_with_params(
                model_name, model_class, params, input_size, output_size, device
            )

            # Load pretrained weights
            model.load_state_dict(torch.load(pretrained_info['model_path'], map_location=device))
            print(f"Loaded pretrained weights for {model_name}")

            # Finetune with smaller learning rate
            finetune_lr = params['learning_rate'] * FINETUNE_LR_FACTOR
            print(f"Finetuning LR: {finetune_lr:.6f} (original: {params['learning_rate']:.6f})")
            
            best_trial = finetune_model(
                model=model,
                train_loader=obs_dataloaders['train'],
                val_loader=obs_dataloaders['val'],
                learning_rate=finetune_lr,
                device=device,
                max_epochs=FINETUNE_MAX_EPOCHS,
                patience=FINETUNE_PATIENCE
            )
            
            # Save finetuned model
            best_trial['params'] = params.copy()
            best_trial['params']['finetune_learning_rate'] = finetune_lr
            best_trial['params']['data_fraction'] = target_fraction
            
            model_path, params_path = save_finetuned_model(
                best_trial, model_name, target_fraction
            )
            
            finetuned_results[model_name] = {
                'model_path': model_path,
                'params_path': params_path,
                'data_fraction': target_fraction,
                'final_val_loss': best_trial['loss']
            }
            
            print(f"{model_name} finetuning completed! Final validation loss: {best_trial['loss']:.6f}")

        except Exception as e:
            print(f"Error finetuning {model_name}: {e}")
            import traceback
            traceback.print_exc()

    return finetuned_results

def main():
    """Main finetuning pipeline for specific fractions"""
    try:
        # Convert single value to list for uniform processing
        if isinstance(TARGET_FRACTIONS, (int, float)):
            fractions_to_process = [TARGET_FRACTIONS]
        else:
            fractions_to_process = TARGET_FRACTIONS
            
        print(f"STEP 2: FINETUNING FOR {fractions_to_process} DATA")
        print("="*70)
        start_time = datetime.now()
        
        # Load pretrained models
        pretrained_models = load_pretrained_models()
        
        if not pretrained_models:
            print("ERROR: No pretrained models found!")
            return
        
        print(f"Found {len(pretrained_models)} pretrained models")
        
        all_results = {}
        
        # Process each fraction
        for fraction in fractions_to_process:
            print(f"\nProcessing fraction: {int(fraction*100)}%")
            
            # Finetune for this fraction
            finetuned_results = finetune_for_fraction(pretrained_models, fraction)
            
            if finetuned_results:
                all_results[fraction] = finetuned_results
                print(f"Completed finetuning for {int(fraction*100)}% data")
            else:
                print(f"ERROR: No models were successfully finetuned for {int(fraction*100)}% data!")
        
        if not all_results:
            print("ERROR: No models were successfully finetuned for any fraction!")
            return
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("FINETUNING COMPLETED!")
        print("="*70)
        
        print(f"Summary:")
        print(f"   Successfully processed fractions: {len(all_results)}")
        print(f"   Total time: {duration}")
        print(f"   Models saved to: {FINETUNED_MODEL_DIR}")
        
        print(f"\nFinal validation losses by fraction:")
        for fraction, results in all_results.items():
            print(f"  {int(fraction*100)}% data:")
            for model_name, info in results.items():
                print(f"     {model_name}: {info['final_val_loss']:.6f}")
        
        print(f"\nFinetuning complete!")
        print(f"Once all fractions are done, run figures_plot.py for analysis.")
        
    except Exception as e:
        print(f"FATAL ERROR in finetuning pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()