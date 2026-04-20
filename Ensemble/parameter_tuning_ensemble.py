# Import libraries
import os
import torch
import numpy as np
import json
import itertools
from datetime import datetime
import torch.nn as nn
from tqdm import tqdm
import copy
import pandas as pd

from depth_wise_ensemble import DepthWiseEnsemble, set_seed
from ensemble_data_processing import DataPreparator, ENSEMBLE_OUTPUT_DIR

# ======= Tuning Parameters =======
# Hyperparameter search space
PARAM_GRID = {
    'learning_rate': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'weight_decay': [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    'scheduler_patience': [2, 3, 5, 10, 20]
}

# Fixed parameters (not part of the search)
FIXED_PARAMS = {
    'early_stopping_patience': 100,
    'init_equal_weights': True,
    'epochs': 400,  
    'batch_size': 32,
    'scheduler_factor': 0.5,
    'min_lr': 1e-4
}

# Maximum number of parameter combinations to try
MAX_COMBINATIONS = 5000

# Data fractions to tune for
TUNE_DATA_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
# ================================================

# Save locations for results and tuning outputs
ENSEMBLE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Ensemble'
RESULTS_DIR = os.path.join(ENSEMBLE_BASE, 'Results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Tuning output will go directly to Ensemble
TUNING_OUTPUT_DIR = ENSEMBLE_BASE
os.makedirs(TUNING_OUTPUT_DIR, exist_ok=True)

class EnsembleTuner:
    """Handles hyperparameter tuning for ensemble models"""
    
    def __init__(self):
        """Initialize the tuner"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_preparator = DataPreparator()
        
        # Define hyperparameter search space
        self.param_grid = PARAM_GRID
        self.fixed_params = FIXED_PARAMS
        
        # Create timestamp for this tuning run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Dictionary to store tuning results
        self.tuning_results = []
    
    def grid_search(self, data_fraction=0.8, use_pretrained=False, max_combinations=MAX_COMBINATIONS):
        """
        Perform grid search to find the best hyperparameters and save the best model
        
        Args:
            data_fraction: Fraction of data to use for tuning
            use_pretrained: Whether to use pretrained models or non-pretrained models
            max_combinations: Maximum number of parameter combinations to try
        
        Returns:
            dict: Best hyperparameters and validation loss
        """
        mode_str = "pretrained" if use_pretrained else "non-pretrained"
        print(f"Starting grid search for {mode_str} models with data fraction {data_fraction}")
        
        # Reset tuning results for this fraction and mode
        self.tuning_results = []
        
        # Generate all possible combinations of hyperparameters
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        # If there are too many combinations, sample a subset
        if len(combinations) > max_combinations:
            print(f"Total combinations: {len(combinations)}, sampling {max_combinations}")
            np.random.seed(42)
            indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]
        
        print(f"Testing {len(combinations)} hyperparameter combinations for {mode_str} models")
        
        # Get ensemble datasets using the specified pretrained mode
        ensemble_loaders, input_size, output_size = self.data_preparator.generate_ensemble_datasets(
            data_fraction=data_fraction,
            use_pretrained=use_pretrained  # Key parameter to distinguish pretrained vs non-pretrained
        )
        
        # Calculate number of models and depths
        num_models, num_depths = self.data_preparator.get_model_info(data_fraction, use_pretrained=use_pretrained)
        
        # Iterate through all combinations
        for i, values in enumerate(combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, values))
            # Add fixed parameters
            params.update(self.fixed_params)
            
            print(f"\nTesting combination {i+1}/{len(combinations)} for {mode_str} models: {params}")
            
            # Create model with current parameters
            model = DepthWiseEnsemble(
                num_models=num_models,
                num_depths=num_depths,
                equal_init=True  # Always use equal initialization
            ).to(self.device)
            
            # Train and evaluate model
            best_val_loss, trained_model = self._train_and_evaluate(
                model, 
                ensemble_loaders,
                params
            )
            
            # Store results
            result = params.copy()
            result['val_loss'] = best_val_loss
            result['model_state'] = copy.deepcopy(trained_model.state_dict())
            self.tuning_results.append(result)
            
            # Save results after each combination
            self._save_tuning_results(data_fraction, use_pretrained)
        
        # Find best hyperparameters
        best_idx = np.argmin([r['val_loss'] for r in self.tuning_results])
        best_result = self.tuning_results[best_idx]
        print(f"\nBest hyperparameters for {mode_str} models: {best_result}")
        
        # Save best parameters and model
        self._save_best_model(best_result, data_fraction, use_pretrained)
        
        return best_result
    
    def _train_and_evaluate(self, model, ensemble_loaders, params):
        """
        Train and evaluate a model with the given parameters
        
        Args:
            model: The model to train
            ensemble_loaders: DataLoaders for training and validation
            params: Training parameters
            
        Returns:
            tuple: (best_val_loss, trained_model)
        """
        # Set up loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=params['scheduler_patience'], 
            factor=params['scheduler_factor'], 
            min_lr=params['min_lr']
        )
        
        # Training parameters
        epochs = params['epochs']  
        best_val_loss = float('inf')
        patience = params['early_stopping_patience']
        patience_counter = 0
        best_model_state = None
        
        # Use tqdm for progress bar instead of printing each epoch
        epoch_iterator = tqdm(range(epochs), desc="Training")
        
        # Training loop
        for epoch in epoch_iterator:
            
            # Training phase
            model.train()
            train_loss = 0.0
            for batch in ensemble_loaders['train']:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(ensemble_loaders['train'])
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in ensemble_loaders['val']:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
            
            val_loss /= len(ensemble_loaders['val'])
            scheduler.step(val_loss)
            
            # Update progress bar description with losses
            epoch_iterator.set_description(f"Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model state
        model.load_state_dict(best_model_state)
        
        # Print final result
        print(f"Final result - Best Val Loss: {best_val_loss:.4f}")
        
        return best_val_loss, model
    
    def _save_tuning_results(self, data_fraction, use_pretrained):
        """Save tuning results to a CSV file (excluding model states)"""
        # Create a copy of results without model states
        results_for_csv = []
        for result in self.tuning_results:
            result_copy = result.copy()
            if 'model_state' in result_copy:
                del result_copy['model_state']
            results_for_csv.append(result_copy)
            
        # Add suffix to distinguish between pretrained and non-pretrained results
        suffix = "_pretrained" if use_pretrained else "_no_pretrained"
        results_path = os.path.join(TUNING_OUTPUT_DIR, f'tuning_results_frac{int(data_fraction*100)}{suffix}_{self.timestamp}.csv')
        
        results_df = pd.DataFrame(results_for_csv)
        results_df.to_csv(results_path, index=False)
        
        mode_str = "pretrained" if use_pretrained else "non-pretrained"
        print(f"Tuning results for {mode_str} models saved to {results_path}")
    
    def _save_best_model(self, best_result, data_fraction, use_pretrained):
        """
        Save best model and parameters from optimization
        
        Args:
            best_result (dict): The best trial from optimization
            data_fraction (float): Data fraction used for tuning
            use_pretrained (bool): Whether pretrained models were used
        """
        # Create paths with suffix to distinguish between pretrained and non-pretrained
        suffix = "_pretrained" if use_pretrained else ""
        
        model_path = os.path.join(RESULTS_DIR, f'best_ensemble_model_frac{int(data_fraction*100)}{suffix}.pth')
        params_path = os.path.join(RESULTS_DIR, f'best_ensemble_params_frac{int(data_fraction*100)}{suffix}.json')
        
        # Get parameters (exclude model_state and val_loss)
        save_params = {k: v for k, v in best_result.items() 
                      if k not in ['model_state', 'val_loss']}
        
        # Add validation loss as info
        save_params['best_val_loss'] = best_result['val_loss']
        save_params['use_pretrained'] = use_pretrained  # Add flag to indicate model type
        
        # Save model state
        torch.save(best_result['model_state'], model_path)
        
        # Save parameters
        with open(params_path, 'w') as f:
            json.dump(save_params, f, indent=4)
        
        mode_str = "pretrained" if use_pretrained else "non-pretrained"
        print(f"Saved best {mode_str} ensemble model and parameters for {int(data_fraction*100)}% data")

def run_tuning_for_both_modes(data_fractions=TUNE_DATA_FRACTIONS, max_combinations=MAX_COMBINATIONS):
    """
    Run hyperparameter tuning process for both pretrained and non-pretrained models
    
    Args:
        data_fractions: List of data fractions to tune for
        max_combinations: Maximum number of parameter combinations to try
    
    Returns:
        dict: Dictionary of best hyperparameters for each data fraction and mode
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Dictionary to store best parameters for each fraction and mode
    best_params_dict = {
        'non_pretrained': {},
        'pretrained': {}
    }
    
    print("Starting comprehensive hyperparameter tuning for both pretrained and non-pretrained models")
    print("="*80)
    
    # Run grid search for each data fraction
    for fraction in data_fractions:
        print(f"\n{'='*60}")
        print(f"TUNING FOR DATA FRACTION: {int(fraction*100)}%")
        print('='*60)
        
        # Create separate tuners for each mode to avoid any interference
        
        # 1. Tune non-pretrained models
        print(f"\n--- TUNING NON-PRETRAINED ENSEMBLE MODELS ---")
        tuner_no_pretrain = EnsembleTuner()
        best_result_no_pretrain = tuner_no_pretrain.grid_search(
            data_fraction=fraction, 
            use_pretrained=False, 
            max_combinations=max_combinations
        )
        best_params_dict['non_pretrained'][fraction] = {k: v for k, v in best_result_no_pretrain.items() 
                                                       if k not in ['model_state']}
        
        # 2. Tune pretrained models
        print(f"\n--- TUNING PRETRAINED ENSEMBLE MODELS ---")
        tuner_pretrain = EnsembleTuner()
        best_result_pretrain = tuner_pretrain.grid_search(
            data_fraction=fraction, 
            use_pretrained=True, 
            max_combinations=max_combinations
        )
        best_params_dict['pretrained'][fraction] = {k: v for k, v in best_result_pretrain.items() 
                                                   if k not in ['model_state']}
    
    # Save summary of all best parameters
    summary_path = os.path.join(RESULTS_DIR, 'tuning_summary_both_modes.json')
    with open(summary_path, 'w') as f:
        json.dump(best_params_dict, f, indent=4, default=str)
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER TUNING COMPLETE FOR ALL DATA FRACTIONS AND BOTH MODES!")
    print(f"Summary saved to: {summary_path}")
    print('='*80)
    
    return best_params_dict

if __name__ == "__main__":
    # Run tuning process for all data fractions and both modes
    best_params_dict = run_tuning_for_both_modes()
    print(f"Hyperparameter tuning complete for all data fractions and both modes!")