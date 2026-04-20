# Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from environment_configuration import setup_environment, get_file_paths # type: ignore

class DataConfig:
    """Configuration for data preprocessing"""
    def __init__(self):
        # Weather features for prediction
        self.weather_features = ['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow']
        # Valid depths (0-20m)
        self.valid_depths = np.arange(0, 20.5, 1)
        # Bathymetry interpolation
        self.depth_range = np.arange(0, 21, dtype=np.float32)

class DataProcessor:
    """Handles data loading and preprocessing"""
    def __init__(self):
        self.config = DataConfig()
        # Setup environment and get paths
        self.base_path = setup_environment()
        self.file_paths = get_file_paths(self.base_path)
        # Process bathymetry data
        self.area_array = self.process_bathymetry()

    def load_lake_data(self):
        """Load data from appropriate location"""
        # Load all required datasets using file paths
        input_data = pd.read_csv(self.file_paths['weather'])
        temp_data = pd.read_csv(self.file_paths['temperature'])
        sim_temp_data = pd.read_csv(self.file_paths['simulated_temperature'])
        ice_data = pd.read_csv(self.file_paths['ice'])
        return input_data, temp_data, sim_temp_data, ice_data

              
    def process_bathymetry(self):
        """
        Process bathymetry data and interpolate areas for all depths using quadratic fitting
        Returns:
        numpy.ndarray: Interpolated areas for depths 0-20m
        """
        # Load bathymetry data
        bathymetry_data = pd.read_csv(self.file_paths['bathymetry'])
            
        # Convert to numpy arrays for processing
        depth = bathymetry_data['Depth(m)'].values
        area = bathymetry_data['Area(m^2)'].values
            
        # Perform quadratic regression (degree=2)
        coeffs = np.polyfit(depth, area, deg=2)
            
        # Interpolate areas for all depths
        interpolated_areas = np.polyval(coeffs, self.config.depth_range)
 
        return interpolated_areas


    def preprocess_data(self, input_data, temp_data, ice_data, is_simulation=False):
        """
        Clean and merge datasets for both real and simulation data
        
        Args:
            input_data (pd.DataFrame): Weather input data
            temp_data (pd.DataFrame): Temperature data (real or simulated)
            ice_data (pd.DataFrame): Ice flag data
            is_simulation (bool): Whether temp_data is simulation data
            
        Returns:
            pd.DataFrame: Processed lake data
        """

        # Convert dates to datetime
        input_data['Date'] = pd.to_datetime(input_data['date'])
        ice_data['Date'] = pd.to_datetime(ice_data['date'])
        
        # Handle temperature data based on type
        if is_simulation:
            # For simulation data
            temp_data['Date'] = pd.to_datetime(temp_data['date'])
            
            # Rename simulation temperature column if needed
            if 'temperature' in temp_data.columns:
                temp_data = temp_data.rename(columns={'temperature': 'wtemp'})
        else:
            # For real observation data
            temp_data['Date'] = pd.to_datetime(temp_data['sampledate'])

        # Merge datasets
        lake_data = pd.merge(input_data, temp_data, on='Date', how='inner')
        lake_data = pd.merge(lake_data, ice_data[['Date', 'ice']], on='Date', how='inner')

        # Remove ice periods and select features
        lake_data = lake_data[lake_data['ice'] == False]
        lake_data = lake_data[['Date', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum',
                              'WindSpeed', 'Rain', 'Snow', 'daynum', 'depth', 'wtemp']]

        return lake_data.dropna()


    def remove_outliers_by_depth(self, data):
        """Remove temperature outliers at each depth using IQR method"""
        def remove_outliers(group, column):
            # Calculate bounds using IQR
            Q1 = group[column].quantile(0.25)
            Q3 = group[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]

        filtered_data = []
        for depth, group in data.groupby('depth'):
            cleaned_group = remove_outliers(group, 'wtemp')
            filtered_data.append(cleaned_group)

        return pd.concat(filtered_data).reset_index(drop=True)

    def prepare_lake_data(self, use_simulation=False):
        """
        Main processing pipeline:
        1. Load raw data
        2. Clean and preprocess
        3. Remove outliers
        4. Create daily profiles
        5. Normalize features
        
        Args:
            use_simulation (bool): If True, use simulation data instead of real observations
            
        Returns: 
            If use_simulation=False: Tuple (normalized_data, raw_data, area_array)
            If use_simulation=True: normalized_data only
        """
        # Load all data
        input_data, temp_data, sim_temp_data, ice_data = self.load_lake_data()
        
        # Preprocess based on data type
        if use_simulation:
            lake_data = self.preprocess_data(input_data, sim_temp_data, ice_data, is_simulation=True)
        else:
            lake_data = self.preprocess_data(input_data, temp_data, ice_data, is_simulation=False)

        # Clean data
        lake_data = lake_data[lake_data['depth'].isin(self.config.valid_depths)]
        lake_data = self.remove_outliers_by_depth(lake_data)

        # Ensure complete depth profiles
        lake_data = lake_data.groupby('Date').filter(
            lambda x: (x['depth'].max() == 20) and (len(x) == 21))

        # Create daily summaries
        weather_day = lake_data.groupby('Date')[self.config.weather_features].mean()
        temp_day = lake_data.groupby('Date')['wtemp'].apply(list)
        daily_data = pd.merge(
            weather_day.reset_index(),
            temp_day.reset_index(name='Temperatures'),
            on='Date'
        )

        # Scale features
        daily_data_scaled = daily_data.copy()
        daily_data_scaled[self.config.weather_features] = MinMaxScaler().fit_transform(
            daily_data_scaled[self.config.weather_features])

        # Return appropriate data based on mode
        if use_simulation:
            return daily_data_scaled
        else:
            return daily_data_scaled, daily_data, self.area_array