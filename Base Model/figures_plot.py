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
GOOGLE_DRIVE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Base Model'
MODEL_OUTPUT_DIR = os.path.join(GOOGLE_DRIVE_BASE, 'Model Output')
HEATMAP_SAVE_DIR = os.path.join(GOOGLE_DRIVE_BASE, 'Figure')
RESULTS_SUMMARY_PATH = os.path.join(GOOGLE_DRIVE_BASE, 'summary_results.csv')

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

def plot_heatmaps():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = SequenceProcessor()
    
    result_table = []

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
    print(f"Fixed test set size (after filtering) = {test_set_size}")

    for model_name, model_class in MODELS.items():
        for frac in DATA_FRACTIONS:
            model_dir = os.path.join(MODEL_OUTPUT_DIR, f"{model_name}_frac{int(frac*100)}")
            param_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

            param_path = os.path.join(model_dir, param_files[0])
            model_path = os.path.join(model_dir, model_files[0])

            with open(param_path, 'r') as f:
                params = json.load(f)

            print(f"Evaluating {model_name} trained with {int(frac*100)}% of data...")
            
            if model_name == 'Transformer':
                cfg = params['transformer_config']
                model = model_class(
                    input_size=input_size_full,
                    d_model=cfg['d_model'],               
                    nhead=cfg['nhead'],                
                    output_size=output_size_full,
                    dropout_rate=params['dropout_rate'],
                    num_layers=params['num_layers'],
                    dim_feedforward_factor=params['dim_feedforward_factor']
                    ).to(device)
            elif model_name == 'CNN_LSTM':
                model = model_class(
                    input_size=input_size_full,
                    hidden_size1=params['hidden_size'],
                    hidden_size2=params['hidden_size'] // 2,
                    hidden_size3=params['hidden_size'] // 4,
                    output_size=output_size_full,
                    dropout_rate=params['dropout_rate'],
                    num_filters_l1=params['num_filters_l1'],
                    num_filters_l2=params['num_filters_l2'],
                    num_filters_l3=params['num_filters_l3'],
                    kernel_size=params['kernel_size']
                ).to(device)
            elif model_name == 'AttentionLSTM':
                model = model_class(
                    input_size=input_size_full,
                    hidden_size1=params['hidden_size'],
                    hidden_size2=params['hidden_size'] // 2,
                    hidden_size3=params['hidden_size'] // 4,
                    output_size=output_size_full,
                    dropout_rate=params['dropout_rate']
                ).to(device)
            else:
                model = model_class(
                    input_size=input_size_full,
                    hidden_size1=params['hidden_size'],
                    hidden_size2=params['hidden_size'] // 2,
                    hidden_size3=params['hidden_size'] // 4,
                    output_size=output_size_full,
                    dropout_rate=params['dropout_rate']
                ).to(device)

            model.load_state_dict(torch.load(model_path))
            print(f"Evaluating on fixed test set with {test_set_size} samples...")
            rmse_list = evaluate_rmse_by_depth(model, fixed_test_loader, device)

            for depth, rmse in enumerate(rmse_list):
                result_table.append({
                    'Model': model_name,
                    'DataFraction': frac,
                    'Depth': depth,
                    'RMSE': rmse
                })

    df = pd.DataFrame(result_table)
    df.to_csv(RESULTS_SUMMARY_PATH, index=False)
    print(f"Saved summary results to {RESULTS_SUMMARY_PATH}")

    # Compute global vmin/vmax for shared colorbar
    vmin = df['RMSE'].min()
    vmax = df['RMSE'].max()

    for model_name in df['Model'].unique():
        print(f"Generating heatmap for {model_name}...")
        subdf = df[df['Model'] == model_name]
        heatmap_data = subdf.pivot(index='DataFraction', columns='Depth', values='RMSE')

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': 'RMSE'}
        )
        plt.ylabel('Training Data Fraction')
        plt.xlabel('Depth (m)')
        plt.tight_layout()

        heatmap_path = os.path.join(HEATMAP_SAVE_DIR, f'{model_name}_heatmap.png')
        plt.savefig(heatmap_path)
        plt.close()
        print(f"Saved heatmap to {heatmap_path}")

if __name__ == "__main__":
    plot_heatmaps()
