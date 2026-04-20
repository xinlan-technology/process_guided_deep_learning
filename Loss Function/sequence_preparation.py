# Import libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import DataProcessor

class SequenceConfig:
    """Configuration for sequence preparation parameters"""
    def __init__(self):
        # Sequence length for input data
        self.sliding_window = 30   
        # Training batch size
        self.batch_size = 32     
        # Weather features used as model inputs  
        self.weather_features = ['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow']
        # Random seed for reproducibility
        self.random_seed = 42      

class SequenceProcessor:
    """Handles sequence creation and data splitting"""
    def __init__(self):
        """Initialize processor with configuration and device setup"""
        self.config = SequenceConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}\n")
        
    def find_valid_sequences(self, datatable):
        """
        Find continuous sequences of days in the dataset
        Args:
            datatable: DataFrame with time series data
        Returns:
            List of valid start dates for continuous sequences
        """
        valid_start_dates = []

        def has_continuous_days(start_index, day_length):
            """Check for continuous days starting from given index"""
            start_date = datatable.iloc[start_index]['Date']
            end_date = start_date + pd.Timedelta(days=(day_length-1))
            expected_dates = pd.date_range(start=start_date, end=end_date)
            actual_dates = datatable.iloc[start_index:start_index + day_length]['Date']
            return all(expected_dates == actual_dates.reset_index(drop=True))

        for idx in range(len(datatable) - (self.config.sliding_window-1)):
            if has_continuous_days(idx, self.config.sliding_window):
                valid_start_dates.append(datatable.iloc[idx]['Date'])

        print(f"Found {len(valid_start_dates)} valid sequences\n")
        return valid_start_dates

    def create_sequences(self, datatable, valid_dates):
        """
        Create input-output sequences from time series data
        Args:
            datatable: DataFrame with time series data
            valid_dates: List of sequence start dates
        Returns:
            Tuple (X_sequences, Y_sequences, daynum_sequences) as numpy arrays
        """
        X_sequences, Y_sequences, daynum_sequences = [], [], []

        for start_date in valid_dates:
            # Get sequence data
            start_idx = datatable[datatable['Date'] == pd.to_datetime(start_date)].index[0]
            
            # Extract weather features for deep learning (7 features)
            X_seq = datatable.iloc[start_idx:start_idx + self.config.sliding_window][self.config.weather_features]
            
            # Extract temperature data
            Y_seq = np.array([
                datatable.iloc[i]['Temperatures'] # Get temperatures at all depths for day i
                for i in range(start_idx, start_idx + self.config.sliding_window)
            ])
            
            # Extract daynum for energy conservation (needed for seasonal adaptive depth)
            # Handle case where daynum column doesn't exist (for simulation data)
            if 'daynum' in datatable.columns:
                daynum_seq = datatable.iloc[start_idx:start_idx + self.config.sliding_window]['daynum'].values
            else:
                # For simulation data, create dummy daynum values (e.g., 1-365)
                # This is just a placeholder since simulation data doesn't need real daynum for energy conservation
                daynum_seq = np.array([1] * self.config.sliding_window)  # Use day 1 as default
            
            X_sequences.append(X_seq.values)
            Y_sequences.append(Y_seq)
            daynum_sequences.append(daynum_seq)

        return np.array(X_sequences), np.array(Y_sequences), np.array(daynum_sequences)

    def prepare_data_splits(self, X_seq, Y_seq, daynum_seq, use_simulation=False, temp_day29=None, weather_day29_30=None, split_ratios=[0.7, 0.2, 0.1]):
        """
        Create train, validation, test splits according to specified ratios
    
        Args:
            X_seq: Scaled input sequence array (7 weather features)
            Y_seq: Scaled target output sequence array
            daynum_seq: Day of year sequence array (for energy conservation)
            use_simulation (bool): Whether we're using simulation data
            temp_day29: Day 29 temperatures (raw) [batch, 21] - for energy storage calculation
            weather_day29_30: Day 29-30 weather features (raw) [batch, 2, 7] - for energy flux calculation
            split_ratios: List of ratios for [train, val, test] (must sum to 1)
        
        Returns:
            Dictionary of DataLoaders for each split
        """
    
        # Compute split sizes
        train_size = int(split_ratios[0] * len(X_seq))
        val_size = int(split_ratios[1] * len(X_seq))

        # Generate random indices and shuffle data
        np.random.seed(self.config.random_seed)
        indices = np.random.permutation(len(X_seq))

        # Split indices into train, validation, and test sets
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        # Create DataLoaders
        dataloaders = {}
        splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }

        for split_name, idx in splits.items():
            # Skip if no data for this split (e.g., no test set)
            if len(idx) == 0:
                continue
            
            # Convert data to PyTorch tensors
            X = torch.tensor(X_seq[idx], dtype=torch.float32).to(self.device)  # Deep learning input: 7 features
            Y = torch.tensor(Y_seq[idx, -1, :], dtype=torch.float32).to(self.device)  # Target: last day temperature (day 30)
            daynum = torch.tensor(daynum_seq[idx, -1], dtype=torch.float32).to(self.device)  # Last day's daynum for energy conservation
        
            # Determine whether to include physics-based inputs
            should_use_physics = False if use_simulation else True
        
            # For observation data with physics constraints
            if should_use_physics:
                # Day 29 temperature data (for energy storage change calculation)
                temp_29 = torch.tensor(temp_day29[idx], dtype=torch.float32).to(self.device)
                # Day 29-30 weather data (for energy flux calculation)
                weather_29_30 = torch.tensor(weather_day29_30[idx], dtype=torch.float32).to(self.device)
            
                # Create dataset with physics constraints: (X, Y, temp_29, weather_29_30, daynum)
                dataset = TensorDataset(X, Y, temp_29, weather_29_30, daynum)
            else:
                # Create dataset for simulation data (no physics constraints): (X, Y)
                dataset = TensorDataset(X, Y)
            
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split_name == 'train'),  # Shuffle only training set
                drop_last=True
            )
            print(f"{split_name} set size: {len(idx)}")

        return dataloaders

    def prepare_sequence_data(self, use_simulation=False, split_ratios=None, data_fraction=1.0):
        """
        Main sequence preparation pipeline for both simulation and observation data
        
        Args:
            use_simulation (bool): If True, prepare simulation data for pretraining
                                  If False, prepare observation data with physics constraints
            split_ratios (list): Optional custom split ratios [train, val, test]
                                Default: [0.8, 0.2, 0.0] for simulation data
                                         [0.7, 0.2, 0.1] for observation data
            data_fraction (float): Fraction of data to use (for testing with smaller datasets)
        
        Returns:
            If use_simulation=True: Tuple (dataloaders, feature_size, target_size)
            If use_simulation=False: Tuple (dataloaders, feature_size, target_size, area_array)
        """
        # Set default split ratios based on data type
        if split_ratios is None:
            split_ratios = [0.8, 0.2, 0.0] if use_simulation else [0.7, 0.2, 0.1]
        
        # Get processed data
        data_processor = DataProcessor()
        
        if use_simulation:
            # Get simulation data (returns only scaled data)
            data_scaled = data_processor.prepare_lake_data(use_simulation=True)
            
            print("\nPreparing simulation data for pretraining")
        else:
            # Get observation data (returns scaled data, raw data, and area array)
            data_scaled, data_raw, area_array = data_processor.prepare_lake_data(use_simulation=False)
            
            print("\nPreparing observation data for training with physics constraints")

        # Identify valid sequences
        valid_dates = self.find_valid_sequences(data_scaled)

        # Create sequences (now returns daynum as well)
        X_scaled, Y_scaled, daynum_seq = self.create_sequences(data_scaled, valid_dates)

        # Shuffle and apply data_fraction
        total = X_scaled.shape[0]
        np.random.seed(self.config.random_seed)
        idx = np.random.permutation(total)
        n_keep = int(total * data_fraction)
        X_scaled = X_scaled[idx[:n_keep]]
        Y_scaled = Y_scaled[idx[:n_keep]]
        daynum_seq = daynum_seq[idx[:n_keep]]
        
        # Handle physics constraints data for observation data
        if not use_simulation and data_raw is not None:
            X_raw, Y_raw, _ = self.create_sequences(data_raw, valid_dates)  # Don't need daynum from raw data

            # Apply same shuffling and data_fraction to raw data
            X_raw = X_raw[idx[:n_keep]]
            Y_raw = Y_raw[idx[:n_keep]]
            
            # Extract data for physics constraints
            # Energy Conservation Logic:
            # - Energy Storage Change: Day 30 (predicted) - Day 29 (actual)
            # - Energy Flux: Average of Day 29 and Day 30 weather
            
            # Day 29 temperature (actual, for energy storage baseline)
            temp_day29 = Y_raw[:, -2, :]  # [batch, 21] - day 29 temperatures (actual values)
            
            # Day 29-30 weather (for energy flux calculation: average of day 29 and day 30)
            weather_day29_30 = X_raw[:, -2:, :]  # [batch, 2, 7] - day 29, 30 weather data
            
            # Generate DataLoaders with scaled inputs, raw environmental data, and daynum
            dataloaders = self.prepare_data_splits(
                X_seq=X_scaled, 
                Y_seq=Y_scaled,
                daynum_seq=daynum_seq,  # Pass daynum for energy conservation
                use_simulation=use_simulation,  
                temp_day29=temp_day29, 
                weather_day29_30=weather_day29_30,
                split_ratios=split_ratios
                )
        else:
            # For simulation data, no physics constraints needed
            dataloaders = self.prepare_data_splits(
                X_seq=X_scaled,
                Y_seq=Y_scaled,
                daynum_seq=daynum_seq,  # Still pass daynum even for simulation
                use_simulation=use_simulation, 
                split_ratios=split_ratios
                )
        
        # Get feature and target sizes
        feature_size = X_scaled.shape[2]  # Number of weather features (7)
        target_size = Y_scaled.shape[2]   # Number of temperature depths (21)
        
        print(f"Deep learning input features: {feature_size}")
        print(f"Temperature prediction depths: {target_size}")
        
        # Return appropriate data based on mode
        if use_simulation:
            return dataloaders, feature_size, target_size
        else:
            return dataloaders, feature_size, target_size, area_array