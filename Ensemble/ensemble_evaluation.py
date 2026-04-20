# Import libraries
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
from depth_wise_ensemble import DepthWiseEnsemble, set_seed
from ensemble_data_processing import DataPreparator, ENSEMBLE_OUTPUT_DIR, DATA_FRACTIONS

# Constants
ENSEMBLE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Ensemble'
RESULTS_DIR = os.path.join(ENSEMBLE_BASE, 'Results')                       
FIGURE_DIR = os.path.join(ENSEMBLE_BASE, 'Figures')                         
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Color scheme for models
MODEL_STYLES = {
    'LSTM': {'color': '#045275', 'marker': 'o', 'linestyle': '-'},
    'Transformer': {'color': '#39B185', 'marker': 's', 'linestyle': '--'},
    'CNN_LSTM': {'color': '#E9E29C', 'marker': '^', 'linestyle': '-.'},
    'AttentionLSTM': {'color': '#EEB479', 'marker': 'D', 'linestyle': ':'},
    'Ensemble': {'color': '#DC3977', 'marker': '*', 'linestyle': '-', 'linewidth': 2.5}
}

class EnsembleEvaluator:
    """Handles evaluation of base models and ensemble models"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.data_preparator = DataPreparator()
        
    def evaluate_models_by_depth(self, data_fractions=DATA_FRACTIONS, use_pretrained=False):
        """
        Evaluate all base models and depth-wise ensemble models on fixed test set, calculating RMSE by depth
        
        Args:
            data_fractions: List of data fractions to evaluate
            use_pretrained: Whether to use pretrained models or non-pretrained models
                              
        Returns:
            pd.DataFrame: DataFrame with evaluation results
        """
        results_table = []
        rmse_by_fraction = {}  # Store RMSE results for each fraction
        
        mode_str = "pretrained" if use_pretrained else "non-pretrained"
        print(f"\n=== Evaluating {mode_str} models ===")
        
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

        # Get all test indices
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
        
        # Evaluate models for each data fraction
        for fraction in data_fractions:
            print(f"\n--- Evaluating {mode_str} models for {int(fraction*100)}% of data ---")
            
            # Load base models with pretrained flag
            base_models = self.data_preparator.load_base_models(fraction, use_pretrained=use_pretrained)
            
            # Use the original test loader for base model evaluation
            test_loader = original_test_loader
            
            # Get predictions from base models
            base_preds = {}
            targets = []
            
            # Collect predictions from base models using original test data
            for batch in tqdm(test_loader, desc=f"Getting {mode_str} base model predictions"):
                # Get input features and target values
                batch_X = batch[0].to(self.device)  # Input features are always first (weather sequence)
                batch_y = batch[1].cpu().numpy()    # Target values are always second (water temps)
                targets.append(batch_y)
                
                for model_name, model in base_models.items():
                    with torch.no_grad():
                        pred = model(batch_X)  # Now batch_X has correct format for base models
                        if model_name not in base_preds:
                            base_preds[model_name] = []
                        base_preds[model_name].append(pred.cpu().numpy())
            
            # Concatenate batches
            targets = np.concatenate(targets, axis=0)
            for model_name in base_preds:
                base_preds[model_name] = np.concatenate(base_preds[model_name], axis=0)
            
            # Create ensemble inputs by concatenating predictions from all base models
            ensemble_inputs = np.concatenate([base_preds[model_name] for model_name in base_preds.keys()], axis=1)
            
            # Load best ensemble model with appropriate suffix
            suffix = "_pretrained" if use_pretrained else ""
            best_model_path = os.path.join(RESULTS_DIR, f'best_ensemble_model_frac{int(fraction*100)}{suffix}.pth')
            
            if os.path.exists(best_model_path):
                print(f"Loading {mode_str} ensemble model from: {best_model_path}")
                # Get model dimensions
                num_models = len(base_models)
                num_depths = output_size
                
                # Create depth-wise ensemble model
                depth_wise_model = DepthWiseEnsemble(
                    num_models=num_models,
                    num_depths=num_depths,
                    equal_init=True
                ).to(self.device)
                
                # Load weights
                depth_wise_model.load_state_dict(torch.load(best_model_path))
                depth_wise_model.eval()
                
                # Get ensemble predictions
                ensemble_inputs_tensor = torch.tensor(ensemble_inputs, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    ensemble_preds = depth_wise_model(ensemble_inputs_tensor).cpu().numpy()
                
                # Add ensemble to predictions dictionary
                base_preds['Ensemble'] = ensemble_preds
                print(f"Models available for plotting: {list(base_preds.keys())}")
            else:
                print(f"Warning: No {mode_str} ensemble model found at {best_model_path}. Skipping ensemble evaluation.")
            
            # Calculate RMSE by depth for each model
            rmse_results = {}
            depths = np.arange(targets.shape[1])  # Get depth values
            
            for model_name in base_preds:
                preds = base_preds[model_name]
                
                # Calculate RMSE for each depth
                rmse_by_depth = np.sqrt(((preds - targets) ** 2).mean(axis=0))
                rmse_results[model_name] = rmse_by_depth
                
                # Add to results table
                for depth, rmse in enumerate(rmse_by_depth):
                    results_table.append({
                        'Model': model_name,
                        'DataFraction': fraction,
                        'Depth': depth,
                        'RMSE': rmse,
                        'ModelType': mode_str  # Add model type for tracking
                    })
                
                # Print average RMSE for debugging
                avg_rmse = rmse_by_depth.mean()
                print(f"Average RMSE for {model_name}: {avg_rmse:.4f}")
            
            # Store RMSE results for this fraction
            rmse_by_fraction[fraction] = {
                'rmse_results': rmse_results,
                'depths': depths
            }
            
            # Plot results for this fraction
            self._plot_rmse_vs_depth(rmse_results, depths, fraction, use_pretrained)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results_table)
        
        # Save results with appropriate suffix
        suffix = "_pretrained" if use_pretrained else ""
        results_path = os.path.join(RESULTS_DIR, f'ensemble_evaluation_results{suffix}.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Saved {mode_str} evaluation results to {results_path}")
        
        # Create average RMSE summary table
        avg_rmse_data = []
        for fraction in sorted(rmse_by_fraction.keys()):
            rmse_results = rmse_by_fraction[fraction]['rmse_results']
            for model_name, rmse_values in rmse_results.items():
                avg_rmse = np.mean(rmse_values)
                avg_rmse_data.append({
                    'Model': model_name,
                    'DataFraction': fraction,
                    'AverageRMSE': avg_rmse,
                    'ModelType': mode_str
                })

        avg_rmse_df = pd.DataFrame(avg_rmse_data)
        avg_rmse_path = os.path.join(RESULTS_DIR, f'average_rmse_summary{suffix}.csv')
        avg_rmse_df.to_csv(avg_rmse_path, index=False)
        print(f"Saved {mode_str} average RMSE summary to {avg_rmse_path}")

        # Print the average RMSE table to console
        print(f"\n{mode_str.title()} Average RMSE Summary:")
        pivot_table = avg_rmse_df.pivot(index='Model', columns='DataFraction', values='AverageRMSE')
        print(pivot_table.round(4))
        
        # Create average RMSE comparison plot
        self._plot_average_rmse_comparison(rmse_by_fraction, use_pretrained)
        
        return results_df
    
    def _plot_rmse_vs_depth(self, rmse_results, depths, data_fraction, use_pretrained=False):
        """
        Plot RMSE vs Depth for all models
        
        Args:
            rmse_results (dict): Dictionary of RMSE values for each model
            depths (np.ndarray): Array of depth values
            data_fraction (float): Data fraction used for training
            use_pretrained (bool): Whether pretrained models were used
        """
        # Create a figure
        plt.figure(figsize=(4.5, 7))
        
        # Plot RMSE vs depth for each model
        for model_name, rmse_values in rmse_results.items():
            style = MODEL_STYLES.get(model_name, {'color': 'gray', 'marker': 'o', 'linestyle': '-'})
            plt.plot(
                rmse_values, 
                depths, 
                label=model_name,
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=style.get('linewidth', 1.5),
                markersize=6
            )
        
        # Set labels (no title)
        plt.xlabel('RMSE', fontsize=12)
        plt.ylabel('Depth (m)', fontsize=12)
        
        # Invert y-axis to have depth 0 at the top
        plt.gca().invert_yaxis()
        
        # Show all depths for better precision
        plt.yticks(depths)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend - smaller font size
        plt.legend(fontsize=9, loc='upper right')
        
        # Set appropriate xlim to show all data clearly
        max_rmse = max([max(rmse) for rmse in rmse_results.values()])
        plt.xlim(0, max_rmse * 1.1)  # Add 10% margin
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure with appropriate suffix
        suffix = "_pretrained" if use_pretrained else "_no_pretrained"
        save_path = os.path.join(FIGURE_DIR, f'rmse_vs_depth_frac{int(data_fraction*100)}{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        mode_str = "pretrained" if use_pretrained else "non-pretrained"
        print(f"Saved {mode_str} RMSE vs Depth plot for {int(data_fraction*100)}% data to {save_path}")
        
        plt.close()
    
    def _plot_average_rmse_comparison(self, rmse_by_fraction, use_pretrained=False):
        """
        Create a bar plot comparing average RMSE across different models and data fractions
        
        Args:
            rmse_by_fraction (dict): Dictionary containing RMSE results for each fraction
            use_pretrained (bool): Whether pretrained models were used
        """
        # Prepare data for plotting
        fractions = []
        model_to_avg_rmse = {}
        
        # Extract data
        for fraction in sorted(rmse_by_fraction.keys()):
            fractions.append(fraction)
            rmse_results = rmse_by_fraction[fraction]['rmse_results']
            
            for model_name, rmse_values in rmse_results.items():
                if model_name not in model_to_avg_rmse:
                    model_to_avg_rmse[model_name] = []
                
                # Calculate average RMSE across all depths
                avg_rmse = np.mean(rmse_values)
                model_to_avg_rmse[model_name].append(avg_rmse)
        
        # Create figure
        plt.figure(figsize=(4.5, 7))
        
        # Set width of bars
        bar_width = 0.15
        index = np.arange(len(fractions))
        
        # Plot bars for each model using the same color scheme
        for i, model_name in enumerate(sorted(model_to_avg_rmse.keys())):
            avg_rmse_list = model_to_avg_rmse[model_name]
            style = MODEL_STYLES.get(model_name, {'color': 'gray'})
            position = index + (i - len(model_to_avg_rmse)/2 + 0.5) * bar_width
            
            plt.bar(
                position, 
                avg_rmse_list, 
                bar_width,
                label=model_name,
                color=style['color'],
                alpha=0.8,
                edgecolor='black',
                linewidth=0.8
            )
        
        # Add labels (no title)
        plt.xlabel('Training Data Fraction', fontsize=12)
        plt.ylabel('Average RMSE', fontsize=12)
        
        # Set x-axis ticks
        plt.xticks(index, [f'{int(frac*100)}%' for frac in fractions], fontsize=10)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Add legend with smaller font
        plt.legend(fontsize=9, loc='upper right')
        
        # Find appropriate y-limit
        y_max = max([max(vals) for vals in model_to_avg_rmse.values()])
        plt.ylim(0, y_max * 1.1)  # Add 10% margin
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure with appropriate suffix
        suffix = "_pretrained" if use_pretrained else "_no_pretrained"
        save_path = os.path.join(FIGURE_DIR, f'average_rmse_comparison{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        mode_str = "pretrained" if use_pretrained else "non-pretrained"
        print(f"Saved {mode_str} average RMSE comparison plot to {save_path}")
        
        plt.close()

    def run_complete_evaluation(self, data_fractions=DATA_FRACTIONS):
        """
        Run complete evaluation for both pretrained and non-pretrained models
        
        Args:
            data_fractions: List of data fractions to evaluate
            
        Returns:
            tuple: (results_no_pretrain, results_pretrain)
        """
        print("Starting complete evaluation for both pretrained and non-pretrained models")
        print("="*80)
        
        # Evaluate non-pretrained models
        print("\n" + "="*50)
        print("PART 1: EVALUATING NON-PRETRAINED MODELS")
        print("="*50)
        results_no_pretrain = self.evaluate_models_by_depth(data_fractions, use_pretrained=False)
        
        # Evaluate pretrained models
        print("\n" + "="*50)
        print("PART 2: EVALUATING PRETRAINED MODELS")
        print("="*50)
        results_pretrain = self.evaluate_models_by_depth(data_fractions, use_pretrained=True)
        
        print("\n" + "="*80)
        print("COMPLETE EVALUATION FINISHED!")
        print("Results saved in:")
        print(f"- Figures: {FIGURE_DIR}")
        print(f"- Results: {RESULTS_DIR}")
        print("="*80)
        
        return results_no_pretrain, results_pretrain
    
if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create evaluator
    evaluator = EnsembleEvaluator()
    
    # Run complete evaluation for both modes (recommended)
    results_no_pretrain, results_pretrain = evaluator.run_complete_evaluation()
    
    print("Ensemble evaluation complete!")