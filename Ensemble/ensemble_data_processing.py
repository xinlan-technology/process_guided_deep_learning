# Import libraries
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json

# Import model classes
from model_architecture import LSTMModel, TransformerModel, CNN_LSTM, AttentionLSTM
from depth_wise_ensemble import DepthWiseEnsemble, set_seed
from sequence_preparation import SequenceProcessor

# Constants - Updated to support both pretrained and non-pretrained models
GOOGLE_DRIVE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Base Model'  
ENSEMBLE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Ensemble'      
PRETRAINED_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Pretraining'    # Path for pretrained models

MODEL_OUTPUT_DIR = os.path.join(GOOGLE_DRIVE_BASE, 'Model Output')          # Non-pretrained models
PRETRAINED_MODEL_DIR = os.path.join(PRETRAINED_BASE, 'Model Output', 'Finetuned_Models')  # Pretrained models
ENSEMBLE_OUTPUT_DIR = os.path.join(ENSEMBLE_BASE, 'Ensemble Output')        
os.makedirs(ENSEMBLE_OUTPUT_DIR, exist_ok=True)

# Dictionary mapping model names to their classes
MODELS = {
    'LSTM': LSTMModel,
    'Transformer': TransformerModel,
    'CNN_LSTM': CNN_LSTM,
    'AttentionLSTM': AttentionLSTM
}

# Standard data fractions
DATA_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

class DataPreparator:
    """Handles loading base models and preparing ensemble datasets"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.processor = SequenceProcessor()
        self.base_models = {}  # Dictionary to store loaded base models with cache key
        
    def load_base_models(self, data_fraction, use_pretrained=False):
        """
        Load the best base models for a specific data fraction
        
        Args:
            data_fraction (float): Fraction of data used for training
            use_pretrained (bool): Whether to load pretrained models or non-pretrained models
            
        Returns:
            dict: Dictionary of loaded models
        """
        # Create cache key to distinguish between pretrained and non-pretrained models
        cache_key = f"{data_fraction}_{use_pretrained}"
        if cache_key in self.base_models:
            return self.base_models[cache_key]
            
        fraction_models = {}
        
        # Set model directory based on pretrained flag
        if use_pretrained:
            base_model_dir = PRETRAINED_MODEL_DIR
            mode_str = "pretrained"
        else:
            base_model_dir = MODEL_OUTPUT_DIR
            mode_str = "non-pretrained"
            
        print(f"\nLoading {mode_str} models trained with {int(data_fraction*100)}% of data...")
        
        # Get dataloader for this data fraction to determine input/output sizes
        dataloaders, input_size, output_size, _ = self.processor.prepare_sequence_data(
            use_simulation=False,
            data_fraction=data_fraction
        )
        
        # Loop through each model type
        for model_name, model_class in MODELS.items():
            try:
                # Construct path to model directory
                model_dir = os.path.join(base_model_dir, f"{model_name}_frac{int(data_fraction*100)}")
                
                # Check if model directory exists
                if not os.path.exists(model_dir):
                    print(f"Warning: Model directory does not exist: {model_dir}")
                    continue
                
                # Find parameter and model files
                param_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                
                if not param_files or not model_files:
                    print(f"Missing parameter or model files for {model_name} at {data_fraction}")
                    continue
                
                # Get paths
                param_path = os.path.join(model_dir, param_files[0])
                model_path = os.path.join(model_dir, model_files[0])
                
                # Load parameters
                with open(param_path, 'r') as f:
                    params = json.load(f)
                
                # Initialize the model with appropriate parameters
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
                    ).to(self.device)
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
                    ).to(self.device)
                elif model_name == 'AttentionLSTM':
                    model = model_class(
                        input_size=input_size,
                        hidden_size1=params['hidden_size'],
                        hidden_size2=params['hidden_size'] // 2,
                        hidden_size3=params['hidden_size'] // 4,
                        output_size=output_size,
                        dropout_rate=params['dropout_rate']
                    ).to(self.device)
                else:  # LSTM
                    model = model_class(
                        input_size=input_size,
                        hidden_size1=params['hidden_size'],
                        hidden_size2=params['hidden_size'] // 2,
                        hidden_size3=params['hidden_size'] // 4,
                        output_size=output_size,
                        dropout_rate=params['dropout_rate']
                    ).to(self.device)
                
                # Load model weights
                model.load_state_dict(torch.load(model_path))
                model.eval()  # Set to evaluation mode
                
                # Store model
                fraction_models[model_name] = model
                print(f"Successfully loaded {model_name} model")
                
            except Exception as e:
                print(f"Error loading {model_name} model: {e}")
        
        # Save models for this data fraction and pretrained mode
        self.base_models[cache_key] = fraction_models
        return fraction_models
    
    def generate_ensemble_datasets(self, data_fraction, use_pretrained=False):
        """
        Generate ensemble datasets by getting predictions from base models
        
        Args:
            data_fraction (float): Fraction of data used for training
            use_pretrained (bool): Whether to use pretrained models or non-pretrained models
                              
        Returns:
            tuple: (dataloaders, input_size, output_size)
        """
        # Load base models (pretrained or non-pretrained based on flag)
        base_models = self.load_base_models(data_fraction, use_pretrained=use_pretrained)
        
        mode_str = "pretrained" if use_pretrained else "non-pretrained"
        print(f"\nGenerating ensemble datasets using {mode_str} models with {int(data_fraction*100)}% training data...")
        
        # Get training/validation data using the specific fraction
        train_val_dataloaders, _, output_size, _ = self.processor.prepare_sequence_data(
            use_simulation=False,
            data_fraction=data_fraction
        )
        
        # Also get fixed test set from 100% data for evaluation
        test_dataloaders, _, _, _ = self.processor.prepare_sequence_data(
            use_simulation=False,
            data_fraction=1.0
        )
        # Rebuild test loader without drop_last to use all test samples
        fixed_test_dataset = test_dataloaders['test'].dataset
        fixed_test_loader = DataLoader(fixed_test_dataset, batch_size=32, shuffle=False, drop_last=False)
        
        # Collections for ensemble data
        ensemble_train_X, ensemble_train_y = [], []
        ensemble_val_X, ensemble_val_y = [], []
        ensemble_test_X, ensemble_test_y = [], []
        
        # Extract data from original loaders
        train_data = [(batch[0], batch[1]) for batch in train_val_dataloaders['train']]
        val_data = [(batch[0], batch[1]) for batch in train_val_dataloaders['val']]
        test_data = [(batch[0], batch[1]) for batch in fixed_test_loader]
        
        print(f"Training data size: {len(train_data)} batches")
        print(f"Validation data size: {len(val_data)} batches")
        print(f"Test data size (from 100% dataset): {len(test_data)} batches")
        
        # Process training data - collect predictions from each base model
        for X_batch, y_batch in tqdm(train_data, desc=f"Processing {int(data_fraction*100)}% training data"):
            # Get base model predictions
            model_outputs = []
            with torch.no_grad():
                for model_name, model in base_models.items():
                    outputs = model(X_batch.to(self.device))
                    model_outputs.append(outputs.cpu())
            
            # Stack predictions as ensemble inputs
            ensemble_inputs = torch.cat(model_outputs, dim=1)
            
            # Store data
            ensemble_train_X.append(ensemble_inputs)
            ensemble_train_y.append(y_batch.cpu())
        
        # Process validation data
        for X_batch, y_batch in tqdm(val_data, desc=f"Processing {int(data_fraction*100)}% validation data"):
            model_outputs = []
            with torch.no_grad():
                for model_name, model in base_models.items():
                    outputs = model(X_batch.to(self.device))
                    model_outputs.append(outputs.cpu())
            
            ensemble_inputs = torch.cat(model_outputs, dim=1)
            ensemble_val_X.append(ensemble_inputs)
            ensemble_val_y.append(y_batch.cpu())
        
        # Process fixed test data (from 100% dataset)
        for X_batch, y_batch in tqdm(test_data, desc="Processing fixed test data (100% dataset)"):
            model_outputs = []
            with torch.no_grad():
                for model_name, model in base_models.items():
                    outputs = model(X_batch.to(self.device))
                    model_outputs.append(outputs.cpu())
            
            ensemble_inputs = torch.cat(model_outputs, dim=1)
            ensemble_test_X.append(ensemble_inputs)
            ensemble_test_y.append(y_batch.cpu())
        
        # Concatenate all batches
        train_X = torch.cat(ensemble_train_X, dim=0) if ensemble_train_X else torch.tensor([])
        train_y = torch.cat(ensemble_train_y, dim=0) if ensemble_train_y else torch.tensor([])
        val_X = torch.cat(ensemble_val_X, dim=0) if ensemble_val_X else torch.tensor([])
        val_y = torch.cat(ensemble_val_y, dim=0) if ensemble_val_y else torch.tensor([])
        test_X = torch.cat(ensemble_test_X, dim=0) if ensemble_test_X else torch.tensor([])
        test_y = torch.cat(ensemble_test_y, dim=0) if ensemble_test_y else torch.tensor([])

        # Remove GLM calibration overlap from test set
        from sequence_preparation import SequenceConfig
        from data_preprocessing import DataProcessor
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
        keep_mask = torch.tensor(~overlap_mask, dtype=torch.bool)
        print(f"Removing {overlap_mask.sum()} overlapping GLM calibration dates from ensemble test set")

        # Filter ensemble test tensors
        test_X = test_X[keep_mask]
        test_y = test_y[keep_mask]

        print(f"Final shapes - Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")
        
        # Create TensorDatasets
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        test_dataset = TensorDataset(test_X, test_y)
        
        # Create DataLoaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create ensemble dataloaders
        ensemble_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        # Get input size (concatenated outputs from all base models)
        # Each base model has output_size predictions, and we have len(models) base models
        input_size = output_size * len(base_models)
        
        return ensemble_loaders, input_size, output_size
    
    def get_model_info(self, data_fraction, use_pretrained=False):
        """
        Get number of models and depths for a specific data fraction
        
        Args:
            data_fraction (float): Fraction of data used
            use_pretrained (bool): Whether to use pretrained models
            
        Returns:
            tuple: (num_models, num_depths)
        """
        # Load base models if not already loaded
        base_models = self.load_base_models(data_fraction, use_pretrained=use_pretrained)
        
        # Get dataloader to determine output size
        _, _, output_size, _ = self.processor.prepare_sequence_data(
            use_simulation=False,
            data_fraction=data_fraction
        )
        
        return len(base_models), output_size