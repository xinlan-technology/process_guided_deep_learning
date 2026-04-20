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

# Constants - Updated to match new basemodel_training.py structure
GOOGLE_DRIVE_BASE = '/content/drive/MyDrive/process_guided_deep_learning/Loss Function'

# Model directories for different training approaches
MODELS_DIR_SCRATCH = '/content/drive/MyDrive/process_guided_deep_learning/Base Model/Model Output'           # Models trained from scratch
MODELS_DIR_PRETRAINED = '/content/drive/MyDrive/process_guided_deep_learning/Pretraining/Model Output/Finetuned_Models'      # Finetuned models

# Dictionary mapping model names to their classes
MODELS = {
    'LSTM': LSTMModel,
    'Transformer': TransformerModel,
    'CNN_LSTM': CNN_LSTM,
    'AttentionLSTM': AttentionLSTM
}

# Define data fractions
DATA_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

class DataPreparator:
    """Handles loading base models and preparing ensemble datasets with support for different training approaches and train sizes"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.processor = SequenceProcessor()
        self.base_models = {}  # Dictionary to store loaded base models
        
    def load_base_models(self, data_fraction, use_finetuned=False):
        """
        Args:
            data_fraction (float): Fraction of data used for training (0.2, 0.4, 0.6, 0.8, 1.0)
            use_finetuned (bool): Whether to load finetuned models (True) or from-scratch models (False)
            
        Returns:
            dict: Dictionary of loaded models
        """
        # Create cache key to distinguish between different training approaches
        cache_key = f"{data_fraction}_{use_finetuned}"
        if cache_key in self.base_models:
            return self.base_models[cache_key]
            
        fraction_models = {}
        
        # Set model directory based on training approach
        if use_finetuned:
            base_model_dir = MODELS_DIR_PRETRAINED  
            mode_str = "finetuned (pretrained)"
        else:
            base_model_dir = MODELS_DIR_SCRATCH     
            mode_str = "from scratch"
            
        print(f"\nLoading {mode_str} models from {base_model_dir}")
        print(f"Training data fraction: {int(data_fraction*100)}%")
        
        # Get dataloader for this data fraction to determine input/output sizes
        dataloaders, input_size, output_size, _ = self.processor.prepare_sequence_data(
            use_simulation=False,
            data_fraction=data_fraction
        )
        
        # Loop through each model type
        for model_name, model_class in MODELS.items():
            try:
                # Construct model directory path
                model_dir = os.path.join(base_model_dir, f"{model_name}_frac{int(data_fraction*100)}")
                
                # Check if model directory exists
                if not os.path.exists(model_dir):
                    print(f"Warning: Model directory does not exist: {model_dir}")
                    continue
                
                # Find parameter and model files
                if use_finetuned:
                    param_files = [f for f in os.listdir(model_dir) if f.startswith('finetuned_params_') and f.endswith('.json')]
                    model_files = [f for f in os.listdir(model_dir) if f.startswith('finetuned_model_') and f.endswith('.pth')]
                else:
                    param_files = [f for f in os.listdir(model_dir) if f.startswith('best_params_') and f.endswith('.json')]
                    model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pth')]
                                
                if not param_files or not model_files:
                    print(f"Missing parameter or model files for {model_name} at fraction {int(data_fraction*100)}%")
                    continue
                
                # Get paths to most recent files
                param_files.sort(reverse=True)
                model_files.sort(reverse=True)
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
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()  # Set to evaluation mode
                
                # Store model
                fraction_models[model_name] = model
                
                # Show training type information if available
                training_info = ""
                if 'training_type' in params:
                    training_info = f" ({params['training_type']})"
                elif 'is_finetuned' in params:
                    training_info = f" ({'finetuned' if params['is_finetuned'] else 'from_scratch'})"
                
                print(f"Successfully loaded {mode_str} {model_name} model{training_info}")
                
            except Exception as e:
                print(f"Error loading {model_name} model: {e}")
                import traceback
                traceback.print_exc()
        
        # Save models for this data fraction, train size, and training approach
        self.base_models[cache_key] = fraction_models
        print(f"Loaded {len(fraction_models)} {mode_str} base models for {int(data_fraction*100)}% data")
        return fraction_models
    
    def generate_ensemble_datasets(self, data_fraction, use_finetuned=False):
        """  
        Args:
            data_fraction (float): Fraction of data used for training (0.2, 0.4, 0.6, 0.8, 1.0)
            use_finetuned (bool): Whether to use finetuned models (True) or from-scratch models (False)
                          
        Returns:
            tuple: (dataloaders, input_size, output_size)
        """
        # Load base models
        base_models = self.load_base_models(data_fraction, use_finetuned=use_finetuned)
        
        mode_str = "finetuned (pretrained)" if use_finetuned else "from scratch"
        print(f"\nGenerating ensemble datasets using {mode_str} models with {int(data_fraction*100)}% training data...")
        
        # Prepare sequence data
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
        
        # Collections for ensemble data (inputs, targets, and energy conservation data)
        ensemble_train_X, ensemble_train_y = [], []
        ensemble_train_temp_day29, ensemble_train_weather_day29_30 = [], []
        ensemble_train_daynum = []
        
        ensemble_val_X, ensemble_val_y = [], []
        ensemble_val_temp_day29, ensemble_val_weather_day29_30 = [], []
        ensemble_val_daynum = []
        
        ensemble_test_X, ensemble_test_y = [], []
        ensemble_test_temp_day29, ensemble_test_weather_day29_30 = [], []
        ensemble_test_daynum = []
        
        # Extract data from original loaders - now including additional tensors for energy conservation
        print(f"Extracting data from dataloaders...")
        train_data = []
        val_data = []
        test_data = []
        
        # Process training data
        for batch in train_val_dataloaders['train']:
            X_batch, y_batch, temp_day29_batch, weather_day29_30_batch, daynum_batch = batch
            train_data.append((
                X_batch, y_batch, temp_day29_batch, weather_day29_30_batch, daynum_batch
            ))
        
        # Process validation data
        for batch in train_val_dataloaders['val']:
            X_batch, y_batch, temp_day29_batch, weather_day29_30_batch, daynum_batch = batch
            val_data.append((
                X_batch, y_batch, temp_day29_batch, weather_day29_30_batch, daynum_batch
            ))
        
        # Process test data
        for batch in fixed_test_loader:
            X_batch, y_batch, temp_day29_batch, weather_day29_30_batch, daynum_batch = batch
            test_data.append((
                X_batch, y_batch, temp_day29_batch, weather_day29_30_batch, daynum_batch
            ))
        
        print(f"Training data size ({int(data_fraction*100)}% data): {len(train_data)} batches")
        print(f"Validation data size: {len(val_data)} batches")
        print(f"Test data size: {len(test_data)} batches")
        
        # Process training data - collect predictions from each base model
        for batch_tuple in tqdm(train_data, desc=f"Processing {int(data_fraction*100)}% training data"):
            X_batch, y_batch, temp_day29, weather_day29_30, daynum_batch = batch_tuple
            
            # Get base model predictions
            model_outputs = []
            with torch.no_grad():
                for model_name, model in base_models.items():
                    outputs = model(X_batch.to(self.device))
                    model_outputs.append(outputs.cpu())
            
            # Stack predictions as ensemble inputs
            ensemble_inputs = torch.cat(model_outputs, dim=1)
            
            # Store data including energy conservation tensors and daynum
            ensemble_train_X.append(ensemble_inputs)
            ensemble_train_y.append(y_batch.cpu())
            ensemble_train_temp_day29.append(temp_day29.cpu())
            ensemble_train_weather_day29_30.append(weather_day29_30.cpu())
            ensemble_train_daynum.append(daynum_batch.cpu())
        
        # Process validation data
        for batch_tuple in tqdm(val_data, desc=f"Processing {int(data_fraction*100)}% validation data"):
            X_batch, y_batch, temp_day29, weather_day29_30, daynum_batch = batch_tuple
            
            model_outputs = []
            with torch.no_grad():
                for model_name, model in base_models.items():
                    outputs = model(X_batch.to(self.device))
                    model_outputs.append(outputs.cpu())
            
            ensemble_inputs = torch.cat(model_outputs, dim=1)
            ensemble_val_X.append(ensemble_inputs)
            ensemble_val_y.append(y_batch.cpu())
            ensemble_val_temp_day29.append(temp_day29.cpu())
            ensemble_val_weather_day29_30.append(weather_day29_30.cpu())
            ensemble_val_daynum.append(daynum_batch.cpu())
        
        # Process fixed test data (from 100% dataset)
        for batch_tuple in tqdm(test_data, desc="Processing fixed test data (100% dataset)"):
            X_batch, y_batch, temp_day29, weather_day29_30, daynum_batch = batch_tuple
            
            model_outputs = []
            with torch.no_grad():
                for model_name, model in base_models.items():
                    outputs = model(X_batch.to(self.device))
                    model_outputs.append(outputs.cpu())
            
            ensemble_inputs = torch.cat(model_outputs, dim=1)
            ensemble_test_X.append(ensemble_inputs)
            ensemble_test_y.append(y_batch.cpu())
            ensemble_test_temp_day29.append(temp_day29.cpu())
            ensemble_test_weather_day29_30.append(weather_day29_30.cpu())
            ensemble_test_daynum.append(daynum_batch.cpu())
        
        # Concatenate all batches
        train_X = torch.cat(ensemble_train_X, dim=0) if ensemble_train_X else torch.tensor([])
        train_y = torch.cat(ensemble_train_y, dim=0) if ensemble_train_y else torch.tensor([])
        train_temp_day29 = torch.cat(ensemble_train_temp_day29, dim=0) if ensemble_train_temp_day29 else torch.tensor([])
        train_weather_day29_30 = torch.cat(ensemble_train_weather_day29_30, dim=0) if ensemble_train_weather_day29_30 else torch.tensor([])
        train_daynum = torch.cat(ensemble_train_daynum, dim=0) if ensemble_train_daynum else torch.tensor([])
        
        val_X = torch.cat(ensemble_val_X, dim=0) if ensemble_val_X else torch.tensor([])
        val_y = torch.cat(ensemble_val_y, dim=0) if ensemble_val_y else torch.tensor([])
        val_temp_day29 = torch.cat(ensemble_val_temp_day29, dim=0) if ensemble_val_temp_day29 else torch.tensor([])
        val_weather_day29_30 = torch.cat(ensemble_val_weather_day29_30, dim=0) if ensemble_val_weather_day29_30 else torch.tensor([])
        val_daynum = torch.cat(ensemble_val_daynum, dim=0) if ensemble_val_daynum else torch.tensor([])
        
        test_X = torch.cat(ensemble_test_X, dim=0) if ensemble_test_X else torch.tensor([])
        test_y = torch.cat(ensemble_test_y, dim=0) if ensemble_test_y else torch.tensor([])
        test_temp_day29 = torch.cat(ensemble_test_temp_day29, dim=0) if ensemble_test_temp_day29 else torch.tensor([])
        test_weather_day29_30 = torch.cat(ensemble_test_weather_day29_30, dim=0) if ensemble_test_weather_day29_30 else torch.tensor([])
        test_daynum = torch.cat(ensemble_test_daynum, dim=0) if ensemble_test_daynum else torch.tensor([])

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

        # Filter all 5 ensemble test tensors
        test_X = test_X[keep_mask]
        test_y = test_y[keep_mask]
        test_temp_day29 = test_temp_day29[keep_mask]
        test_weather_day29_30 = test_weather_day29_30[keep_mask]
        test_daynum = test_daynum[keep_mask]

        print(f"Final shapes - Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")
        
        # Create TensorDatasets with all 5 tensors for energy conservation and daynum
        train_dataset = TensorDataset(train_X, train_y, train_temp_day29, train_weather_day29_30, train_daynum)
        val_dataset = TensorDataset(val_X, val_y, val_temp_day29, val_weather_day29_30, val_daynum)
        test_dataset = TensorDataset(test_X, test_y, test_temp_day29, test_weather_day29_30, test_daynum)
        
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
    
    def get_model_info(self, data_fraction, use_finetuned=False):
        """
        Args:
            data_fraction (float): Fraction of data used (0.2, 0.4, 0.6, 0.8, 1.0)
            use_finetuned (bool): Whether to use finetuned models
            
        Returns:
            tuple: (num_models, num_depths)
        """
        # Load base models if not already loaded
        base_models = self.load_base_models(data_fraction, use_finetuned=use_finetuned)
        
        # Get dataloader to determine output size
        _, _, output_size, _ = self.processor.prepare_sequence_data(
            use_simulation=False,
            data_fraction=data_fraction
        )
        
        return len(base_models), output_size