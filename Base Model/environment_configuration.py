import os
from google.colab import drive # type: ignore

def setup_environment():
    """
    Setup environment and mount Google Drive.
    Returns:
        str: Base path to the data directory
    Raises:
        RuntimeError: If data directory is not found after mounting
    """
    # Define the base path for data
    base_path = "/content/drive/MyDrive/Process Guided Deep Learning Data/Lake Mendota Observation Data"
    
    # If path doesn't exist, try mounting Google Drive
    if not os.path.exists(base_path):
        drive.mount('/content/drive')
        
        # Check path again after mounting
        if not os.path.exists(base_path):
            raise RuntimeError(f"Data directory not found at: {base_path}")
    
    return base_path

def get_file_paths(base_path):
    """
    Generate paths for required data files.
    Args:
        base_path (str): Base directory path containing the data files
    Returns:
        dict: Dictionary containing paths for each data file
    """
    file_paths = {
        'weather': os.path.join(base_path, "Lake_Mendota_Weather_Input.csv"),
        'temperature': os.path.join(base_path, "Lake_Mendota_Water_Temperature.csv"),
        'ice': os.path.join(base_path, "Lake_Mendota_Ice_Flag.csv"),
        'bathymetry': os.path.join(base_path, "Lake_Mendota_Bathymetry.csv"),
        'simulated_temperature': os.path.join(base_path, "Simulate_Temp_Mendota.csv"),
        'glm_calibration': "/content/drive/MyDrive/General Lake Model Simulation/field_temp_oxy.csv"
    }
    return file_paths