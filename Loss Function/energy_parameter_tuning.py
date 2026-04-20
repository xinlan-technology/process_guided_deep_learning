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
import hashlib
import random

from depth_wise_ensemble import DepthWiseEnsemble
from ensemble_data_processing import DataPreparator
from ensemble_energy_conservation import EnsembleEnergyConservation

# Hyperparameter search space
PARAM_GRID = {
    'energy_weight': [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    'threshold': [30, 40, 45, 50, 55, 60, 70, 100, 150]
}

# Fixed parameters (not part of the search)
FIXED_PARAMS = {
    'early_stopping_patience': 100,
    'epochs': 400,
    'batch_size': 32,
    'scheduler_factor': 0.5,
    'min_lr': 1e-4
}

# Define data fractions
DATA_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Stage 2 results storage path
ENSEMBLE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Loss Function'
FIGURE_2_RESULTS = '/content/drive/MyDrive/process_guided_deep_learning/Ensemble/Results'
RESULTS_DIR = os.path.join(ENSEMBLE_BASE, 'Results')
ENERGY_TUNING_DIR = os.path.join(ENSEMBLE_BASE, 'Energy_Tuning')

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ENERGY_TUNING_DIR, exist_ok=True)

def set_seed_comprehensive(seed):
    """Comprehensive seed setting for complete determinism"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_deterministic_seed(base_seed, params):
    """Generate deterministic seed using hash function"""
    seed_string = f"{base_seed}_{params['learning_rate']}_{params['energy_weight']}_{params['threshold']}"
    hash_object = hashlib.md5(seed_string.encode())
    hash_hex = hash_object.hexdigest()
    unique_seed = int(hash_hex[:8], 16) % (2**31 - 1)
    return unique_seed

class UnifiedEnergyTuner:
    """Unified tuner for both ensemble and energy parameters"""
    
    def __init__(self):
        """Initialize the tuner"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_preparator = DataPreparator()
        self.energy_calculator = EnsembleEnergyConservation(device=self.device)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tuning_results = []
    
    def tune_all_parameters(self, data_fraction=1.0, use_finetuned=False):
        """
        Args:
            data_fraction (float): Fraction of data to use (0.2, 0.4, 0.6, 0.8, 1.0)
            use_finetuned (bool): Whether to use finetuned models (True) or from-scratch models (False)
        """
        mode_str = "finetuned (pretrained)" if use_finetuned else "from scratch"
        print(f"Starting unified parameter tuning: {mode_str} models, {int(data_fraction*100)}% data")
        
        # Generate ensemble datasets
        ensemble_loaders, input_size, output_size = self.data_preparator.generate_ensemble_datasets(
            data_fraction=data_fraction,
            use_finetuned=use_finetuned
        )
        num_models, num_depths = self.data_preparator.get_model_info(
            data_fraction, use_finetuned=use_finetuned
        )

        # Load best ensemble model and params from Ensemble
        suffix = "_pretrained" if use_finetuned else ""
        fraction_str = f"frac{int(data_fraction*100)}"
        
        fig2_model_path = os.path.join(FIGURE_2_RESULTS, f'best_ensemble_model_{fraction_str}{suffix}.pth')
        fig2_params_path = os.path.join(FIGURE_2_RESULTS, f'best_ensemble_params_{fraction_str}{suffix}.json')
        
        # Load Ensemble best parameters (lr, weight_decay, etc.)
        with open(fig2_params_path, 'r') as f:
            fig2_params = json.load(f)
        
        # Load Ensemble best model weights
        fig2_model_state = torch.load(fig2_model_path)
        
        print(f"Loaded Ensemble ensemble: lr={fig2_params['learning_rate']}, "
              f"weight_decay={fig2_params['weight_decay']}")
        
        # Generate parameter combinations
        param_names = list(PARAM_GRID.keys())
        param_values = list(PARAM_GRID.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(combinations)} parameter combinations")
        
        # Reset tuning results
        self.tuning_results = []
        best_overall_loss = float('inf')
        best_overall_result = None
        
        for i, values in enumerate(combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, values))
            params.update(FIXED_PARAMS)
            params['learning_rate'] = fig2_params['learning_rate']
            params['weight_decay'] = fig2_params['weight_decay']
            params['scheduler_patience'] = fig2_params['scheduler_patience']
            
            print(f"[{i+1}/{len(combinations)}] lr={params['learning_rate']}, "
                  f"E={params['energy_weight']}, T={params['threshold']}")
            
            try:
                # Create model and load Ensemble best weights
                model = DepthWiseEnsemble(
                    num_models=num_models,
                    num_depths=num_depths,
                    equal_init=True
                ).to(self.device)
                model.load_state_dict(fig2_model_state)
                
                # Train model
                val_loss, trained_model = self._train_with_all_constraints(
                    model, ensemble_loaders, params, use_finetuned
                )
                
                # Store results
                result = params.copy()
                result['val_loss'] = val_loss
                result['use_finetuned'] = use_finetuned
                result['training_failed'] = False
                self.tuning_results.append(result)
                
                # Update best overall result
                if val_loss < best_overall_loss:
                    best_overall_loss = val_loss
                    best_overall_result = {
                        'model': copy.deepcopy(trained_model),
                        'params': params.copy(),
                        'val_loss': val_loss
                    }
                
                print(f"Val loss: {val_loss:.6f}")
                
            except Exception as e:
                print(f"Training failed: {e}")
                
                # Store failed result
                result = params.copy()
                result['val_loss'] = float('inf')
                result['use_finetuned'] = use_finetuned
                result['training_failed'] = True
                self.tuning_results.append(result)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if best_overall_result is None:
            raise RuntimeError("All training runs failed!")
        
        print(f"\nBest overall combination:")
        print(f"lr: {best_overall_result['params']['learning_rate']}, "
              f"E: {best_overall_result['params']['energy_weight']}, "
              f"T: {best_overall_result['params']['threshold']}")
        print(f"Best overall val loss: {best_overall_result['val_loss']:.6f}")
        
        # Save tuning results
        self._save_tuning_results(best_overall_result, data_fraction, use_finetuned)
        
        return best_overall_result['params']
    
    def _train_with_all_constraints(self, model, ensemble_loaders, params, use_finetuned):
        """Train model with energy conservation constraints"""
        
        # Set deterministic seed
        unique_seed = generate_deterministic_seed(42, params)
        set_seed_comprehensive(unique_seed)
        
        train_loader = ensemble_loaders['train']
        val_loader = ensemble_loaders['val']
        
        # Set up optimizer and loss functions
        mse_criterion = nn.MSELoss()
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
        
        # Training control variables
        epochs = params['epochs']
        best_val_loss = float('inf')
        patience = params['early_stopping_patience']
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_mse = 0.0
            train_energy = 0.0
            
            for batch in train_loader:
                x, y, temp_day29, weather_day29_30, daynum = [item.to(self.device) for item in batch]
                
                optimizer.zero_grad()
                outputs = model(x)
                
                mse_loss = mse_criterion(outputs, y)
                
                energy_loss = self.energy_calculator.calculate_energy_loss(
                    pred_temps=outputs,
                    temp_day29=temp_day29,
                    weather_day29_30=weather_day29_30,
                    daynum=daynum,
                    threshold=params['threshold']
                )
                
                total_loss = mse_loss + params['energy_weight'] * energy_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                train_mse += mse_loss.item()
                train_energy += energy_loss.item()
            
            # Average training losses
            train_loss /= len(train_loader)
            train_mse /= len(train_loader)
            train_energy /= len(train_loader)
            
            # Validation phase (MSE only)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = model(x)
                    loss = mse_criterion(outputs, y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return best_val_loss, model
    
    def _save_tuning_results(self, best_overall_result, data_fraction, use_finetuned):
        """
        Save tuning results
        """
        suffix = "_pretrained" if use_finetuned else ""
        fraction_str = f"frac{int(data_fraction*100)}"
        
        # Define file paths
        unified_model_path = os.path.join(ENERGY_TUNING_DIR, f'best_unified_model_{fraction_str}{suffix}.pth')
        unified_params_path = os.path.join(ENERGY_TUNING_DIR, f'best_unified_params_{fraction_str}{suffix}.json')
        
        results_path = os.path.join(ENERGY_TUNING_DIR, f'unified_tuning_results_{fraction_str}{suffix}_{self.timestamp}.csv')
        
        try:
            # Save overall best (unified) model
            torch.save(best_overall_result['model'].state_dict(), unified_model_path)
            
            # Save overall best parameters
            save_params_unified = {k: v for k, v in best_overall_result['params'].items()}
            save_params_unified['best_val_loss'] = best_overall_result['val_loss']
            save_params_unified['use_finetuned'] = use_finetuned
            save_params_unified['data_fraction'] = data_fraction
            save_params_unified['model_type'] = 'unified_best'
            
            with open(unified_params_path, 'w') as f:
                json.dump(save_params_unified, f, indent=4)
            
            print(f"Saved unified (overall best) model to {unified_model_path}")
            
            # Save detailed tuning results
            results_df = pd.DataFrame(self.tuning_results)
            results_df.to_csv(results_path, index=False)
            
            print(f"Saved detailed results to {results_path}")
            
        except Exception as e:
            print(f"Error saving: {e}")
            raise

def run_unified_tuning(data_fractions=DATA_FRACTIONS, test_both_modes=False):
    """ 
    Args:
        data_fractions (list): List of data fractions to test [0.2, 0.4, 0.6, 0.8, 1.0]
        test_both_modes (bool): Whether to test both finetuned and from-scratch modes
    """
    set_seed_comprehensive(42)
    
    tuner = UnifiedEnergyTuner()
    
    all_results = {
        'from_scratch': {},
        'finetuned': {}
    }
    
    print("Starting Energy Conservation Tuning - Different Data Fractions")
    print("Loading ensemble baselines from Ensemble, fine-tuning with energy constraints")
    print("Will save: best_unified_model_*.pth (best energy-constrained model)")
    
    # Loop through different data fractions
    for fraction in data_fractions:
        print(f"\n" + "="*80)
        print(f"PROCESSING {int(fraction*100)}% DATA")
        print("="*80)
        
        # From-scratch model tuning
        try:
            best_params_scratch = tuner.tune_all_parameters(
                data_fraction=fraction, 
                use_finetuned=False
            )
            all_results['from_scratch'][f"{fraction}"] = best_params_scratch
        except Exception as e:
            print(f"Error in from-scratch tuning: {e}")
            all_results['from_scratch'][f"{fraction}"] = None
        
        # Finetuned model tuning (if needed)
        if test_both_modes:
            try:
                best_params_finetuned = tuner.tune_all_parameters(
                    data_fraction=fraction, 
                    use_finetuned=True
                )
                all_results['finetuned'][f"{fraction}"] = best_params_finetuned
            except Exception as e:
                print(f"Error in finetuned tuning: {e}")
                all_results['finetuned'][f"{fraction}"] = None
    
    # Save summary
    summary_path = os.path.join(ENERGY_TUNING_DIR, f'unified_summary_{tuner.timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
    
    print(f"\nTuning complete! Summary: {summary_path}")
    
    return all_results

if __name__ == "__main__":
    
    # Run tuning for all training sizes
    results = run_unified_tuning(test_both_modes=True)
    
    print("All parameter tuning completed!")