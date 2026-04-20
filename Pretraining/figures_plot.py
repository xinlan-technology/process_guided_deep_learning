# Import libraries
import os
import torch
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from model_architecture import LSTMModel, TransformerModel, CNN_LSTM, AttentionLSTM
from data_preprocessing import DataProcessor
from sequence_preparation import SequenceProcessor

# Constants
GOOGLE_DRIVE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Pretraining'
FIGURE1_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Base Model'

# Model paths
FINETUNED_MODEL_DIR = os.path.join(GOOGLE_DRIVE_BASE, 'Model Output', 'Finetuned_Models')
NO_PRETRAIN_MODEL_DIR = os.path.join(FIGURE1_BASE, 'Model Output')

# Save paths
HEATMAP_SAVE_DIR = os.path.join(GOOGLE_DRIVE_BASE, 'Improvement_Heatmaps')
RESULTS_SUMMARY_PATH = os.path.join(GOOGLE_DRIVE_BASE, 'improvement_results.csv')

# Dictionary mapping model names to their classes
MODELS = {
    'LSTM': LSTMModel,
    'Transformer': TransformerModel,
    'CNN_LSTM': CNN_LSTM,
    'AttentionLSTM': AttentionLSTM
}

DATA_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

os.makedirs(HEATMAP_SAVE_DIR, exist_ok=True)

def evaluate_rmse_by_depth(model, test_loader, device):
    """Evaluate model and return RMSE by depth"""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_X, batch_y = batch[0].to(device), batch[1].to(device)
            pred = model(batch_X)
            if pred is None or pred.shape[0] == 0:
                continue
            preds.append(pred.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    rmse_by_depth = np.sqrt(((preds - targets) ** 2).mean(axis=0))
    return rmse_by_depth.tolist()

def load_and_evaluate_model(model_name, model_path, params_path, test_loader, input_size, output_size, device):
    """Load model and evaluate performance"""
    model_class = MODELS[model_name]
    
    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Create model
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
    
    # Load weights and evaluate
    model.load_state_dict(torch.load(model_path, map_location=device))
    rmse_by_depth = evaluate_rmse_by_depth(model, test_loader, device)
    
    return rmse_by_depth

def find_model_files(model_dir):
    """Find model and parameter files in directory"""
    if not os.path.exists(model_dir):
        return None, None
    
    param_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not param_files or not model_files:
        return None, None
    
    return os.path.join(model_dir, model_files[0]), os.path.join(model_dir, param_files[0])

def calculate_improvements():
    """Calculate improvement of pretrained models over no-pretrain models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = SequenceProcessor()
    
    print("Creating fixed test dataset from full data (100%)...")
    dataloaders_full, input_size_full, output_size_full, area_array = processor.prepare_sequence_data(
        use_simulation=False,
        data_fraction=1.0,
        split_ratios=[0.7, 0.2, 0.1]
    )

    # Remove GLM calibration overlap from test set
    from sequence_preparation import SequenceConfig
    from torch.utils.data import DataLoader, TensorDataset
    config = SequenceConfig()
    data_processor = DataProcessor()
    data_scaled, _, _ = data_processor.prepare_lake_data(use_simulation=False)

    # Find valid sequences (same logic as SequenceProcessor)
    valid_start_dates = []
    for idx in range(len(data_scaled) - (config.sliding_window - 1)):
        start_date = data_scaled.iloc[idx]['Date']
        end_date = start_date + pd.Timedelta(days=(config.sliding_window - 1))
        expected_dates = pd.date_range(start=start_date, end=end_date)
        actual_dates = data_scaled.iloc[idx:idx + config.sliding_window]['Date']
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
    original_test_dataset = dataloaders_full['test'].dataset
    filtered_tensors = [t[keep_mask[:len(t)]] for t in original_test_dataset.tensors]
    filtered_dataset = TensorDataset(*filtered_tensors)
    fixed_test_loader = DataLoader(filtered_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    test_set_size = len(fixed_test_loader.dataset)
    print(f"Fixed test set size (after filtering) = {test_set_size}\n")

    improvement_results = []

    for model_name in MODELS.keys():
        print(f"Processing {model_name}...")
        
        for frac in DATA_FRACTIONS:
            print(f"  Data fraction: {int(frac*100)}%")
            
            # Find pretrained+finetuned model
            finetuned_dir = os.path.join(FINETUNED_MODEL_DIR, f"{model_name}_frac{int(frac*100)}")
            finetuned_model_path, finetuned_params_path = find_model_files(finetuned_dir)
            
            # Find no-pretrain model
            no_pretrain_dir = os.path.join(NO_PRETRAIN_MODEL_DIR, f"{model_name}_frac{int(frac*100)}")
            no_pretrain_model_path, no_pretrain_params_path = find_model_files(no_pretrain_dir)
            
            if (finetuned_model_path is None or no_pretrain_model_path is None):
                print(f"    WARNING: Missing models for {model_name} at {int(frac*100)}%")
                continue
            
            try:
                # Evaluate both models
                print(f"    Evaluating pretrained+finetuned model...")
                finetuned_rmse = load_and_evaluate_model(
                    model_name, finetuned_model_path, finetuned_params_path,
                    fixed_test_loader, input_size_full, output_size_full, device
                )
                
                print(f"    Evaluating no-pretrain model...")
                no_pretrain_rmse = load_and_evaluate_model(
                    model_name, no_pretrain_model_path, no_pretrain_params_path,
                    fixed_test_loader, input_size_full, output_size_full, device
                )
                
                # Calculate improvement (positive means pretrained is better, negative means worse)
                for depth, (no_pre_rmse, pre_rmse) in enumerate(zip(no_pretrain_rmse, finetuned_rmse)):
                    improvement = no_pre_rmse - pre_rmse  # Positive = improvement, Negative = degradation
                    
                    improvement_results.append({
                        'Model': model_name,
                        'DataFraction': frac,
                        'Depth': depth,
                        'No_Pretrain_RMSE': no_pre_rmse,
                        'Pretrained_RMSE': pre_rmse,
                        'RMSE_Improvement': improvement
                    })
                
                avg_improvement = np.mean([no_pre - pre for no_pre, pre in zip(no_pretrain_rmse, finetuned_rmse)])
                print(f"    Average RMSE improvement: {avg_improvement:.4f}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
        
        print()

    return pd.DataFrame(improvement_results)

def plot_improvement_heatmaps(df):
    """Generate clean improvement effect heatmaps with consistent color scale"""
    print("Generating improvement heatmaps...")
    
    # Save results
    df.to_csv(RESULTS_SUMMARY_PATH, index=False)
    print(f"Saved improvement results to {RESULTS_SUMMARY_PATH}")
    
    # Set clean style
    plt.style.use('default')
    sns.set_context("paper", font_scale=1.1)
    
    print("Creating RMSE improvement heatmaps...")
    
    # Calculate global color range for ALL models together
    global_vmax = max(abs(df['RMSE_Improvement'].min()), abs(df['RMSE_Improvement'].max()))
    global_vmin = -global_vmax
    
    print(f"Using consistent color range: [{global_vmin:.3f}, {global_vmax:.3f}]")
    
    for model_name in df['Model'].unique():
        model_data = df[df['Model'] == model_name]
        
        # Convert DataFraction to percentage strings BEFORE pivoting
        model_data_copy = model_data.copy()
        model_data_copy['DataFraction'] = model_data_copy['DataFraction'].apply(lambda x: f'{int(x*100)}%')
        
        heatmap_data = model_data_copy.pivot(index='DataFraction', columns='Depth', values='RMSE_Improvement')
        
        # Reorder rows to ensure correct order (20%, 40%, 60%, 80%, 100%)
        fraction_order = ['20%', '40%', '60%', '80%', '100%']
        heatmap_data = heatmap_data.reindex(fraction_order)
        
        # Create figure with clean layout
        plt.figure(figsize=(12, 6))
        
        # Create heatmap with GLOBAL color range for consistency
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            vmin=global_vmin,  # Use global range
            vmax=global_vmax,  # Use global range
            cbar_kws={'label': 'RMSE Change'},
            linewidths=0.3,
            square=False,
            annot_kws={'size': 9}
        )
        
        # Clean labels without title
        plt.ylabel('Training Data Fraction', fontsize=12)
        plt.xlabel('Depth (m)', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save with model name in filename
        save_path = os.path.join(HEATMAP_SAVE_DIR, f'{model_name}_improvement.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved {model_name} improvement heatmap")

def generate_summary_statistics(df):
    """Generate concise summary statistics"""
    print("\nSummary Statistics:")
    print("=" * 40)
    
    for model_name in df['Model'].unique():
        model_data = df[df['Model'] == model_name]
        
        print(f"\n{model_name}:")
        print(f"  Average improvement: {model_data['RMSE_Improvement'].mean():.2f}")
        print(f"  Best improvement: {model_data['RMSE_Improvement'].max():.2f}")
        print(f"  Worst case: {model_data['RMSE_Improvement'].min():.2f}")
        
        # Find best case
        best_idx = model_data['RMSE_Improvement'].idxmax()
        best_frac = model_data.loc[best_idx, 'DataFraction']
        best_depth = model_data.loc[best_idx, 'Depth']
        print(f"  Best case: {int(best_frac*100)}% data, depth {best_depth}m")

def main():
    """Main function"""
    print("Starting Pretraining Improvement Analysis...")
    print("=" * 50)
    
    try:
        # Calculate improvements
        df_improvements = calculate_improvements()
        
        if df_improvements.empty:
            print("ERROR: No improvement data calculated. Check model paths.")
            return
        
        # Generate clean heatmaps
        plot_improvement_heatmaps(df_improvements)
        
        # Generate summary statistics
        generate_summary_statistics(df_improvements)
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETED!")
        print("=" * 50)
        print(f"Heatmaps saved to: {HEATMAP_SAVE_DIR}")
        print(f"Data saved to: {RESULTS_SUMMARY_PATH}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()