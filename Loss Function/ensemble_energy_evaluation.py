# Import libraries
import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
from depth_wise_ensemble import DepthWiseEnsemble, set_seed
from ensemble_data_processing import DataPreparator

# Data fractions
DATA_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

# Stage 2 results directory
ENSEMBLE_BASE_STAGE2 = '/content/drive/MyDrive/process_guided_deep_learning/Loss Function'
STAGE2_ENERGY_DIR = os.path.join(ENSEMBLE_BASE_STAGE2, 'Energy_Tuning')
FIGURE_2_RESULTS = '/content/drive/MyDrive/process_guided_deep_learning/Ensemble/Results'

# Figure and results directories
FIGURE_DIR = os.path.join(ENSEMBLE_BASE_STAGE2, 'Evaluation_Figures')
RESULTS_DIR = os.path.join(ENSEMBLE_BASE_STAGE2, 'Evaluation_Results')

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Color scheme for different ensemble types
ENSEMBLE_STYLES = {
    'Ensemble': {'color': '#045275', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.0},
    'Ensemble + Energy': {'color': '#EEB479', 'marker': '^', 'linestyle': '-.', 'linewidth': 2.0},
    'Ensemble + Pretraining': {'color': '#39B185', 'marker': 's', 'linestyle': '--', 'linewidth': 2.0},
    'Ensemble + Energy + Pretraining': {'color': '#DC3977', 'marker': 'D', 'linestyle': ':', 'linewidth': 2.5}
}

class EnsembleEnergyEvaluator:
    """Evaluates different ensemble configurations with energy constraints"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.data_preparator = DataPreparator()
        
    def get_ensemble_model_path(self, data_fraction, use_finetuned=False):
        """
        Args:
            data_fraction: Data fraction used (0.2, 0.4, 0.6, 0.8, 1.0)
            use_finetuned: Whether to use finetuned models
            
        Returns:
            str: Path to the ensemble model, or None if not found
        """
        suffix = "_pretrained" if use_finetuned else ""
        model_path = os.path.join(FIGURE_2_RESULTS, f'best_ensemble_model_frac{int(data_fraction*100)}{suffix}.pth')
        
        if os.path.exists(model_path):
            print(f"  Found ensemble model (weight=0): {model_path}")
            return model_path
        else:
            print(f"  Ensemble model not found: {model_path}")
            return None
    
    def get_unified_model_path(self, data_fraction, use_finetuned=False):
        """
        Args:
            data_fraction: Data fraction used (0.2, 0.4, 0.6, 0.8, 1.0)
            use_finetuned: Whether to use finetuned models
            
        Returns:
            str: Path to the unified model, or None if not found
        """
        suffix = "_pretrained" if use_finetuned else ""
        model_path = os.path.join(STAGE2_ENERGY_DIR, f'best_unified_model_frac{int(data_fraction*100)}{suffix}.pth')
        
        if os.path.exists(model_path):
            print(f"  Found unified model: {model_path}")
            return model_path
        else:
            print(f"  Unified model not found: {model_path}")
            return None
    
    def evaluate_all_ensemble_types(self, data_fractions=DATA_FRACTIONS):
        """
        Args:
            data_fractions: List of data fractions to evaluate [0.2, 0.4, 0.6, 0.8, 1.0]
            
        Returns:
            pd.DataFrame: Evaluation results
        """
        results_table = []
        rmse_by_fraction = {}
        
        # Get original test dataset from full data (100%)
        print("\nPreparing original test dataset from full data (100%)...")
        original_dataloaders, _, output_size, _ = self.data_preparator.processor.prepare_sequence_data(
            use_simulation=False,
            data_fraction=1.0
        )

        # Remove GLM calibration overlap from test set
        from sequence_preparation import SequenceConfig
        from data_preprocessing import DataProcessor
        from torch.utils.data import DataLoader, TensorDataset
        config = SequenceConfig()
        data_processor = DataProcessor()
        data_scaled, _, _ = data_processor.prepare_lake_data(use_simulation=False)

        # Find valid sequences (same logic as SequenceProcessor)
        valid_start_dates = []
        for i in range(len(data_scaled) - (config.sliding_window - 1)):
            start_date = data_scaled.iloc[i]['Date']
            end_date = start_date + pd.Timedelta(days=(config.sliding_window - 1))
            expected_dates = pd.date_range(start=start_date, end=end_date)
            actual_dates = data_scaled.iloc[i:i + config.sliding_window]['Date']
            if all(expected_dates == actual_dates.reset_index(drop=True)):
                valid_start_dates.append(start_date)

        # First shuffle (same as prepare_sequence_data with data_fraction=1.0)
        total = len(valid_start_dates)
        np.random.seed(config.random_seed)
        idx_shuffle1 = np.random.permutation(total)
        valid_dates_shuffled = np.array(valid_start_dates)[idx_shuffle1]

        # Second shuffle and split (same as prepare_data_splits with [0.7, 0.2, 0.1])
        n = len(valid_dates_shuffled)
        train_size = int(0.7 * n)
        val_size = int(0.2 * n)

        np.random.seed(config.random_seed)
        idx_shuffle2 = np.random.permutation(n)

        # Get all test indices (drop_last handled by DataLoader later)
        test_idx = idx_shuffle2[train_size + val_size:]

        # Compute target dates for each test sequence
        test_start_dates = valid_dates_shuffled[test_idx]
        test_target_dates = pd.to_datetime([d + pd.Timedelta(days=config.sliding_window - 1) for d in test_start_dates])

        # Load GLM calibration dates and find overlap
        field_data = pd.read_csv(data_processor.file_paths['glm_calibration'])
        field_dates = pd.to_datetime(field_data['datetime'].str[:10]).unique()

        overlap_mask = test_target_dates.isin(field_dates)
        keep_mask = ~overlap_mask
        print(f"Removing {overlap_mask.sum()} overlapping GLM calibration dates from test set")

        # Filter test dataset tensors and rebuild DataLoader
        original_test_dataset = original_dataloaders['test'].dataset
        filtered_tensors = [t[keep_mask[:len(t)]] for t in original_test_dataset.tensors]
        filtered_dataset = TensorDataset(*filtered_tensors)
        original_test_loader = DataLoader(filtered_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

        if not original_test_loader:
            print("Error: No test loader found in original dataloaders")
            return pd.DataFrame()

        print(f"Original test set size (after filtering) = {len(original_test_loader.dataset)}")
        
        # Loop through different data fractions
        for fraction in data_fractions:
            print(f"\n" + "="*80)
            print(f"EVALUATING MODELS TRAINED WITH {int(fraction*100)}% DATA")
            print("="*80)
            
            # Store ensemble results
            ensemble_results = {}

            # 1. Ensemble (weight=0, no pretraining) - Best pure ensemble
            print(f"Loading Ensemble model (weight=0, best validation)...")
            base_predictions_no_pretrain, targets = self._get_base_model_predictions(
                fraction, original_test_loader, use_finetuned=False
            )

            # Use best_ensemble_model (weight=0 best model)
            ensemble_results['Ensemble'] = self._load_and_evaluate_ensemble_model(
                fraction=fraction, use_finetuned=False,
                base_predictions=base_predictions_no_pretrain, targets=targets, output_size=output_size
            )

            # 2. Ensemble + Energy (weight>0, no pretraining) - Best with energy constraints
            print(f"Loading Ensemble + Energy model...")
            ensemble_results['Ensemble + Energy'] = self._load_and_evaluate_unified_model(
                fraction=fraction, use_finetuned=False,
                base_predictions=base_predictions_no_pretrain, targets=targets, output_size=output_size
            )

            # 3. Ensemble + Pretraining (weight=0, with pretraining) - Best pretrained pure ensemble
            print(f"Loading Ensemble + Pretraining model...")
            base_predictions_pretrain, _ = self._get_base_model_predictions(
                fraction, original_test_loader, use_finetuned=True
            )

            ensemble_results['Ensemble + Pretraining'] = self._load_and_evaluate_ensemble_model(
                fraction=fraction, use_finetuned=True,
                base_predictions=base_predictions_pretrain, targets=targets, output_size=output_size
            )

            # 4. Ensemble + Energy + Pretraining (weight>0, with pretraining) - Best with both
            print(f"Loading Ensemble + Energy + Pretraining model...")
            ensemble_results['Ensemble + Energy + Pretraining'] = self._load_and_evaluate_unified_model(
                fraction=fraction, use_finetuned=True,
                base_predictions=base_predictions_pretrain, targets=targets, output_size=output_size
            )
            
            # Calculate RMSE by depth for each ensemble type
            rmse_results = {}
            depths = np.arange(targets.shape[1])
            
            for ensemble_name, predictions in ensemble_results.items():
                if predictions is not None:
                    # Calculate RMSE for each depth
                    rmse_by_depth = np.sqrt(((predictions - targets) ** 2).mean(axis=0))
                    rmse_results[ensemble_name] = rmse_by_depth
                    
                    # Add to results table
                    for depth, rmse in enumerate(rmse_by_depth):
                        results_table.append({
                            'EnsembleType': ensemble_name,
                            'DataFraction': fraction,
                            'Depth': depth,
                            'RMSE': rmse
                        })
                    
                    # Print average RMSE
                    avg_rmse = rmse_by_depth.mean()
                    print(f"Average RMSE for {ensemble_name}: {avg_rmse:.4f}")
                else:
                    print(f"Warning: {ensemble_name} model not found for fraction {int(fraction*100)}%")
            
            # Store results for plotting
            rmse_by_fraction[f"{fraction}"] = {
                'rmse_results': rmse_results,
                'depths': depths,
                'fraction': fraction
            }
            
            # Plot results for this fraction
            if rmse_results:  # Only plot if we have results
                self._plot_ensemble_rmse_vs_depth(rmse_results, depths, fraction)
                
        # Plot average RMSE comparison (bar chart)
        if rmse_by_fraction:
            self._plot_average_rmse_comparison(rmse_by_fraction)
                
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results_table)
        results_path = os.path.join(RESULTS_DIR, 'ensemble_energy_evaluation_results_all_fractions.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Saved evaluation results to {results_path}")
        
        return results_df
    
    def _get_base_model_predictions(self, fraction, test_loader, use_finetuned=False):
        """
        Args:
            fraction: Data fraction (0.2, 0.4, 0.6, 0.8, 1.0)
            test_loader: Test data loader
            use_finetuned: Whether to use finetuned base models
            
        Returns:
            tuple: (base_predictions, targets)
        """
        base_models = self.data_preparator.load_base_models(fraction, use_finetuned=use_finetuned)
        
        print(f"  Loaded {len(base_models)} {'finetuned' if use_finetuned else 'non-finetuned'} base models ({int(fraction*100)}% data)")
        
        base_preds = {}
        targets = []
        
        # Get predictions from each base model
        for batch in tqdm(test_loader, desc=f"Getting {'finetuned' if use_finetuned else 'non-finetuned'} base model predictions ({int(fraction*100)}% data)"):
            batch_X = batch[0].to(self.device)
            batch_y = batch[1].cpu().numpy()
            targets.append(batch_y)
            
            for model_name, model in base_models.items():
                with torch.no_grad():
                    pred = model(batch_X)
                    if model_name not in base_preds:
                        base_preds[model_name] = []
                    base_preds[model_name].append(pred.cpu().numpy())
        
        # Concatenate batches
        targets = np.concatenate(targets, axis=0)
        for model_name in base_preds:
            base_preds[model_name] = np.concatenate(base_preds[model_name], axis=0)
        
        # Create ensemble input by concatenating all base model predictions
        ensemble_input = np.concatenate([base_preds[model_name] for model_name in base_preds.keys()], axis=1)
        
        print(f"  Generated ensemble input shape: {ensemble_input.shape}")
        return ensemble_input, targets
    
    def _load_and_evaluate_ensemble_model(self, fraction, use_finetuned, 
                                         base_predictions, targets, output_size):
        """
        Args:
            fraction: Data fraction (0.2, 0.4, 0.6, 0.8, 1.0)
            use_finetuned: Whether to use finetuned base models
            base_predictions: Predictions from base models
            targets: Target values
            output_size: Output size of each base model
            
        Returns:
            numpy.ndarray: ensemble predictions, or None if model not found
        """
        model_path = self.get_ensemble_model_path(fraction, use_finetuned)
        
        if model_path is None:
            print(f"  Ensemble model (weight=0) not found for {int(fraction*100)}% data")
            return None
        
        # Load model
        num_models = base_predictions.shape[1] // output_size  # Calculate number of base models
        
        ensemble_model = DepthWiseEnsemble(
            num_models=num_models,
            num_depths=output_size,
            equal_init=True
        ).to(self.device)
        
        try:
            ensemble_model.load_state_dict(torch.load(model_path, map_location=self.device))
            ensemble_model.eval()
            print(f"  Model loaded successfully (num_models={num_models}, num_depths={output_size})")
        except Exception as e:
            print(f"  Failed to load model: {e}")
            return None
        
        # Get predictions
        ensemble_input_tensor = torch.tensor(base_predictions, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            ensemble_preds = ensemble_model(ensemble_input_tensor).cpu().numpy()
        
        print(f"  Predictions generated: shape {ensemble_preds.shape}")
        return ensemble_preds
    
    def _load_and_evaluate_unified_model(self, fraction, use_finetuned, 
                                       base_predictions, targets, output_size):
        """
        Args:
            fraction: Data fraction (0.2, 0.4, 0.6, 0.8, 1.0)
            use_finetuned: Whether to use finetuned base models
            base_predictions: Predictions from base models
            targets: Target values
            output_size: Output size of each base model
            
        Returns:
            numpy.ndarray: ensemble predictions, or None if model not found
        """
        model_path = self.get_unified_model_path(fraction, use_finetuned)
        
        if model_path is None:
            print(f"  Unified model not found for {int(fraction*100)}% data")
            return None
        
        # Load model
        num_models = base_predictions.shape[1] // output_size  # Calculate number of base models
        
        ensemble_model = DepthWiseEnsemble(
            num_models=num_models,
            num_depths=output_size,
            equal_init=True
        ).to(self.device)
        
        try:
            ensemble_model.load_state_dict(torch.load(model_path, map_location=self.device))
            ensemble_model.eval()
            print(f"  Model loaded successfully (num_models={num_models}, num_depths={output_size})")
        except Exception as e:
            print(f"  Failed to load model: {e}")
            return None
        
        # Get predictions
        ensemble_input_tensor = torch.tensor(base_predictions, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            ensemble_preds = ensemble_model(ensemble_input_tensor).cpu().numpy()
        
        print(f"  Predictions generated: shape {ensemble_preds.shape}")
        return ensemble_preds
    
    def _plot_ensemble_rmse_vs_depth(self, rmse_results, depths, data_fraction):
        """
        Args:
            rmse_results: Dictionary of RMSE values for each ensemble type
            depths: Array of depth values
            data_fraction: Data fraction used (0.2, 0.4, 0.6, 0.8, 1.0)
            
        Returns:
            None
        """
        plt.figure(figsize=(4.5, 7))
        
        # Plot RMSE vs depth for each ensemble type in specific order
        ensemble_order = ['Ensemble', 'Ensemble + Energy', 'Ensemble + Pretraining', 'Ensemble + Energy + Pretraining']
        
        for ensemble_name in ensemble_order:
            if ensemble_name in rmse_results:
                rmse_values = rmse_results[ensemble_name]
                style = ENSEMBLE_STYLES.get(ensemble_name, {'color': 'gray', 'marker': 'o', 'linestyle': '-'})
                plt.plot(
                    rmse_values, 
                    depths, 
                    label=ensemble_name,
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    markersize=6
                )
        
        # Set labels
        plt.xlabel('RMSE', fontsize=12)
        plt.ylabel('Depth (m)', fontsize=12)
        
        # Invert y-axis (surface at top)
        plt.gca().invert_yaxis()
        
        # Show all depths
        plt.yticks(depths)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend with smaller font
        plt.legend(fontsize=8, loc='upper right')
        
        # Calculate optimized x-axis range for this specific plot
        all_rmse_values = []
        for rmse_values in rmse_results.values():
            all_rmse_values.extend(rmse_values)
        
        if all_rmse_values:
            max_rmse = max(all_rmse_values)
            plt.xlim(0, max_rmse * 1.1)  # Add 10% margin
            print(f"  X-axis range for {int(data_fraction*100)}% data: 0 to {max_rmse * 1.1:.4f}")
        else:
            # Fallback in case no data
            plt.xlim(0, 1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(FIGURE_DIR, f'ensemble_comparison_rmse_vs_depth_frac{int(data_fraction*100)}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ensemble comparison RMSE vs Depth plot for {int(data_fraction*100)}% data to {save_path}")
        
        plt.close()
    
    def _plot_average_rmse_comparison(self, rmse_by_fraction):
        """
        Create a bar plot comparing average RMSE across different ensemble types and data fractions
        
        Args:
            rmse_by_fraction (dict): Dictionary containing RMSE results for each fraction
        """
        # Prepare data for plotting
        fractions = []
        ensemble_to_avg_rmse = {}
        
        # Extract data
        for fraction_key in sorted(rmse_by_fraction.keys(), key=lambda x: float(x)):
            fraction = rmse_by_fraction[fraction_key]['fraction']
            fractions.append(fraction)
            rmse_results = rmse_by_fraction[fraction_key]['rmse_results']
            
            for ensemble_name, rmse_values in rmse_results.items():
                if ensemble_name not in ensemble_to_avg_rmse:
                    ensemble_to_avg_rmse[ensemble_name] = []
                
                # Calculate average RMSE across all depths
                avg_rmse = np.mean(rmse_values)
                ensemble_to_avg_rmse[ensemble_name].append(avg_rmse)
        
        # Create figure
        plt.figure(figsize=(4.5, 7))
        
        # Set width of bars
        bar_width = 0.15
        index = np.arange(len(fractions))
        
        # Define ensemble order for consistent plotting
        ensemble_order = ['Ensemble', 'Ensemble + Energy', 'Ensemble + Pretraining', 'Ensemble + Energy + Pretraining']
        
        # Plot bars for each ensemble type using the same color scheme
        for i, ensemble_name in enumerate(ensemble_order):
            if ensemble_name in ensemble_to_avg_rmse:
                avg_rmse_list = ensemble_to_avg_rmse[ensemble_name]
                style = ENSEMBLE_STYLES.get(ensemble_name, {'color': 'gray'})
                position = index + (i - len(ensemble_order)/2 + 0.5) * bar_width
                
                plt.bar(
                    position, 
                    avg_rmse_list, 
                    bar_width,
                    label=ensemble_name,
                    color=style['color'],
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.8
                )
        
        # Add labels
        plt.xlabel('Training Data Fraction', fontsize=12)
        plt.ylabel('Average RMSE', fontsize=12)
        
        # Set x-axis ticks
        plt.xticks(index, [f'{int(frac*100)}%' for frac in fractions], fontsize=10)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add legend with smaller font
        plt.legend(fontsize=8, loc='upper right')
        
        # Find appropriate y-limit
        all_vals = []
        for vals in ensemble_to_avg_rmse.values():
            all_vals.extend(vals)
        if all_vals:
            y_max = max(all_vals)
            plt.ylim(0, y_max * 1.1)  # Add 10% margin
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(FIGURE_DIR, f'average_rmse_comparison_ensemble_types.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved average RMSE comparison plot to {save_path}")
        
        plt.close()

def run_ensemble_energy_evaluation(data_fractions=DATA_FRACTIONS):
    """
    Args:
        data_fractions: List of data fractions to evaluate [0.2, 0.4, 0.6, 0.8, 1.0]
        
    Returns:
        pd.DataFrame: Evaluation results
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create evaluator
    evaluator = EnsembleEnergyEvaluator()
    
    print("="*80)
    print("ENSEMBLE COMPARISON EVALUATION - DIFFERENT DATA FRACTIONS")
    print("="*80)
    print("This will evaluate 4 types of ensemble models for different data fractions:")
    print("1. Ensemble (best weight=0 model, no pretraining)")
    print("2. Ensemble + Energy (best unified model, no pretraining)")
    print("3. Ensemble + Pretraining (best weight=0 model, with pretraining)")
    print("4. Ensemble + Energy + Pretraining (best unified model, with pretraining)")
    print(f"Data fractions: {data_fractions}")
    print("="*80)
    
    # Run evaluation
    results = evaluator.evaluate_all_ensemble_types(data_fractions)
    
    print("\n" + "="*80)
    print("ENSEMBLE COMPARISON EVALUATION COMPLETE!")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Figures saved in: {FIGURE_DIR}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    # Run the evaluation for all data fractions
    results = run_ensemble_energy_evaluation()
        
    print("Ensemble comparison evaluation complete!")