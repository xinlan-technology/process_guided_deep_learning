# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from environment_configuration import setup_environment, get_file_paths

def calculate_vapour_pressure_saturated(temp):
    """Calculate saturated vapor pressure in millibars"""
    exponent = (9.28603523 - (2332.37885/(temp+273.15))) * tf.math.log(10.0)
    return tf.math.exp(exponent)

def calculate_vapour_pressure_air(rel_hum, temp):
    """Calculate vapor pressure"""
    rh_scaling_factor = tf.constant(1.0, dtype=tf.float32)
    rel_hum_tensor = tf.cast(rel_hum, dtype=tf.float32)
    return rh_scaling_factor * (rel_hum_tensor / 100.0) * calculate_vapour_pressure_saturated(temp)

def calculate_air_density(air_temp, rh):
    """Calculate air density in kg/m^3"""
    mwrw2a = tf.constant(18.016 / 28.966, dtype=tf.float32)
    c_gas = tf.constant(1.0e3 * 8.31436 / 28.966, dtype=tf.float32)
    p = tf.constant(1013.0, dtype=tf.float32)
    
    air_temp = tf.cast(air_temp, dtype=tf.float32)
    rh = tf.cast(rh, dtype=tf.float32)
    
    vapPressure = calculate_vapour_pressure_air(rh, air_temp)
    r = mwrw2a * vapPressure / (p - vapPressure)
    density = (1.0 / c_gas * (1 + r) / (1 + r / mwrw2a) * p / (air_temp + 273.15)) * 100
    return density

def calculate_wind_speed_10m(ws, ref_height=2.):
    """Calculate wind speed at 10m height"""
    c_z0 = tf.constant(0.001, dtype=tf.float32)
    height_10m = tf.constant(10.0, dtype=tf.float32)
    
    ws = tf.cast(ws, dtype=tf.float32)
    ref_height = tf.cast(ref_height, dtype=tf.float32)
    
    wind_speed_10m = ws * (tf.math.log(height_10m / c_z0) / tf.math.log(ref_height / c_z0))
    return wind_speed_10m

def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):
    """Calculate latent heat flux"""
    surf_temp = tf.cast(surf_temp, dtype=tf.float32)
    air_temp = tf.cast(air_temp, dtype=tf.float32)
    rel_hum = tf.cast(rel_hum, dtype=tf.float32)
    wind_speed = tf.cast(wind_speed, dtype=tf.float32)
    
    rho_a = calculate_air_density(air_temp, rel_hum)
    c_E = tf.constant(0.0013, dtype=tf.float32)
    lambda_v = tf.constant(2.453e6, dtype=tf.float32)
    omega = tf.constant(0.622, dtype=tf.float32)
    p = tf.constant(1013.0, dtype=tf.float32)
    
    U_10 = calculate_wind_speed_10m(wind_speed)
    e_s = calculate_vapour_pressure_saturated(surf_temp)
    e_a = calculate_vapour_pressure_air(rel_hum, air_temp)
    
    latent_heat_flux = -rho_a * c_E * lambda_v * U_10 * (omega / p) * (e_s - e_a)
    return latent_heat_flux

def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):
    """Calculate sensible heat flux"""
    surf_temp = tf.cast(surf_temp, dtype=tf.float32)
    air_temp = tf.cast(air_temp, dtype=tf.float32)
    rel_hum = tf.cast(rel_hum, dtype=tf.float32)
    wind_speed = tf.cast(wind_speed, dtype=tf.float32)
    
    rho_a = calculate_air_density(air_temp, rel_hum)
    c_a = tf.constant(1005.0, dtype=tf.float32)
    c_H = tf.constant(0.0013, dtype=tf.float32)
    
    U_10 = calculate_wind_speed_10m(wind_speed)
    sensible_heat_flux = -rho_a * c_a * c_H * U_10 * (surf_temp - air_temp)
    return sensible_heat_flux

def calculate_energy_fluxes(R_sw_arr, R_lw_arr, air_temp, rel_hum, ws, surf_temps):
    """Calculate total energy fluxes"""
    # Cast inputs to tensors
    R_sw_arr = tf.cast(R_sw_arr, dtype=tf.float32)
    R_lw_arr = tf.cast(R_lw_arr, dtype=tf.float32)
    air_temp = tf.cast(air_temp, dtype=tf.float32)
    rel_hum = tf.cast(rel_hum, dtype=tf.float32)
    ws = tf.cast(ws, dtype=tf.float32)
    surf_temps = tf.cast(surf_temps, dtype=tf.float32)
    
    # Constants
    e_s = tf.constant(0.985, dtype=tf.float32)
    alpha_sw = tf.constant(0.07, dtype=tf.float32)
    alpha_lw = tf.constant(0.03, dtype=tf.float32)
    sigma = tf.constant(5.67e-8, dtype=tf.float32)
    
    # Calculate outgoing longwave radiation
    R_lw_out_arr = e_s * sigma * tf.pow(surf_temps + 273.15, 4)
    
    # Calculate mean values for flux calculations
    R_sw_arr_mean = (R_sw_arr[..., :-1] + R_sw_arr[..., 1:]) / 2
    R_lw_arr_mean = (R_lw_arr[..., :-1] + R_lw_arr[..., 1:]) / 2
    R_lw_out_arr_mean = (R_lw_out_arr[..., :-1] + R_lw_out_arr[..., 1:]) / 2
    
    # Calculate latent and sensible heat fluxes
    E = (calculate_heat_flux_latent(surf_temps[..., :-1], air_temp[..., :-1], rel_hum[..., :-1], ws[..., :-1]) * 0.5 +
         calculate_heat_flux_latent(surf_temps[..., 1:], air_temp[..., 1:], rel_hum[..., 1:], ws[..., 1:]) * 0.5)
    H = (calculate_heat_flux_sensible(surf_temps[..., :-1], air_temp[..., :-1], rel_hum[..., :-1], ws[..., :-1]) * 0.5 +
         calculate_heat_flux_sensible(surf_temps[..., 1:], air_temp[..., 1:], rel_hum[..., 1:], ws[..., 1:]) * 0.5)
    
    # Total energy flux
    fluxes = (R_sw_arr_mean * (1 - alpha_sw) +
              R_lw_arr_mean * (1 - alpha_lw) -
              R_lw_out_arr_mean + E + H)
    
    return fluxes

def calculate_density(temp):
    """Calculate water density based on temperature"""
    temp = tf.cast(temp, dtype=tf.float32)
    return 1000 * (1 - (temp + 288.9414) * (temp - 3.9863)**2 / (508929.2 * (temp + 68.12963)))

def tf_polyfit(x, y, degree):
    """Linear regression using TensorFlow"""
    X = tf.stack([tf.pow(x, i) for i in range(degree, -1, -1)], axis=1)
    Y = tf.expand_dims(y, axis=1)
    B = tf.linalg.inv(tf.matmul(tf.transpose(X), X))
    B = tf.matmul(B, tf.matmul(tf.transpose(X), Y))
    return tf.squeeze(B)

def remove_outliers(datatable, column_name):
    """Remove outliers using IQR method"""
    Q1 = datatable[column_name].quantile(0.25)
    Q3 = datatable[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return datatable[(datatable[column_name] >= lower_bound) & (datatable[column_name] <= upper_bound)]

def calculate_full_depth_energy(temps, densities, area_array, unit_depth=1.0):
    """
    Calculate lake energy using full depth (0-20m)
    
    Args:
        temps: Temperature tensor [batch_size, 21] (depths 0-20m)
        densities: Density tensor [batch_size, 21]
        area_array: Area array [21] for depths 0-20m
        unit_depth: Unit depth value (default 1.0m)
    
    Returns:
        Energy tensor [batch_size]
    """
    temps = tf.cast(temps, dtype=tf.float32)
    densities = tf.cast(densities, dtype=tf.float32)
    area_array = tf.cast(area_array, dtype=tf.float32)
    unit_depth = tf.cast(unit_depth, dtype=tf.float32)
    c_water = tf.constant(4186.0, dtype=tf.float32)
    
    # Calculate energy for all depths (0-20m)
    energy = tf.reduce_sum(
        temps * densities * area_array * unit_depth * c_water,
        axis=-1
    )
    return energy

def segment_continuous_dates_within_period(period_data):
    """Find continuous date segments within a period"""
    period_data = period_data.sort_values('Date').copy()
    period_data['date_diff'] = period_data['Date'].diff().dt.days
    period_data['segment'] = (period_data['date_diff'] > 1).cumsum()
    
    continuous_segments = []
    for _, segment in period_data.groupby('segment'):
        if len(segment) > 2:  # Need at least 3 points for energy change calculation
            continuous_segments.append(segment)
    return continuous_segments

def calculate_energy_balance_for_segment(segment, area_array):
    """
    Calculate energy flux and storage change for a continuous segment
    
    Args:
        segment: DataFrame with continuous dates
        area_array: Area array for depth integration
    
    Returns:
        tuple: (energy_fluxes, energy_storage_changes) as numpy arrays
    """
    if len(segment) <= 1:
        return None, None
    
    try:
        # Prepare input data
        weather_data = tf.constant(segment[['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed']].values, dtype=tf.float32)
        temp_data = tf.constant(segment['Temperatures'].tolist(), dtype=tf.float32)
        surf_temps = temp_data[:, 0]  # Surface temperature (depth 0)
        
        # Calculate energy fluxes
        energy_fluxes = calculate_energy_fluxes(
            weather_data[:, 0], weather_data[:, 1], weather_data[:, 2],
            weather_data[:, 3], weather_data[:, 4], surf_temps
        )
        
        # Calculate water densities
        densities = tf.map_fn(calculate_density, temp_data)
        
        # Calculate lake energies using full depth
        surface_area = area_array[0]
        
        energies = []
        for i in range(temp_data.shape[0]):
            energy = calculate_full_depth_energy(
                temp_data[i:i+1], densities[i:i+1], area_array, unit_depth=1.0
            )
            energies.append(energy[0])
        
        energies = tf.stack(energies)
        
        # Calculate energy storage change (normalized by surface area and time)
        energy_storage_change = energies[1:] - energies[:-1]
        energy_storage_change = energy_storage_change / (86400 * surface_area)  # W/m²
        
        return energy_fluxes.numpy(), energy_storage_change.numpy()
        
    except Exception as e:
        print(f"  Error processing segment: {e}")
        return None, None

def analyze_monthly_energy_balance(monthly_segments, area_array):
    """
    Analyze energy balance for each month using full depth
    
    Args:
        monthly_segments: List of (month_name, month_data) tuples
        area_array: Area array for depth integration
    
    Returns:
        Dictionary with monthly results
    """
    monthly_results = {}
    
    for month_name, month_data in monthly_segments:
        print(f"\nProcessing {month_name}: {len(month_data)} data points")
        
        # Find continuous segments
        continuous_segments = segment_continuous_dates_within_period(month_data)
        if not continuous_segments:
            print(f"  No continuous segments found for {month_name}")
            continue
        
        print(f"  Found {len(continuous_segments)} continuous segments")
        
        # Collect all flux and storage data for this month
        all_flux_values = []
        all_storage_changes = []
        
        for segment in continuous_segments:
            flux_values, storage_changes = calculate_energy_balance_for_segment(segment, area_array)
            
            if flux_values is not None and storage_changes is not None:
                all_flux_values.extend(flux_values.tolist())
                all_storage_changes.extend(storage_changes.tolist())
        
        if all_flux_values and all_storage_changes:
            # Convert to numpy arrays for easier manipulation
            flux_array = np.array(all_flux_values)
            storage_array = np.array(all_storage_changes)
            
            # Check for extreme outliers (values beyond reasonable physical limits)
            print(f"  Before outlier removal: {len(flux_array)} points")
            print(f"  Flux range: [{flux_array.min():.1f}, {flux_array.max():.1f}] W/m²")
            print(f"  Storage range: [{storage_array.min():.1f}, {storage_array.max():.1f}] W/m²")
            
            # Define reasonable physical limits for energy flux and storage change
            max_reasonable_value = 500  # W/m²
            min_reasonable_value = -500  # W/m²
            
            # Find indices of reasonable values
            flux_reasonable = (flux_array >= min_reasonable_value) & (flux_array <= max_reasonable_value)
            storage_reasonable = (storage_array >= min_reasonable_value) & (storage_array <= max_reasonable_value)
            reasonable_indices = flux_reasonable & storage_reasonable
            
            # Filter out extreme outliers
            flux_filtered = flux_array[reasonable_indices]
            storage_filtered = storage_array[reasonable_indices]
            
            if len(flux_filtered) < len(flux_array):
                removed_count = len(flux_array) - len(flux_filtered)
                print(f"  Removed {removed_count} extreme outliers (>{max_reasonable_value} or <{min_reasonable_value} W/m²)")
                print(f"  After outlier removal: {len(flux_filtered)} points")
                print(f"  New flux range: [{flux_filtered.min():.1f}, {flux_filtered.max():.1f}] W/m²")
                print(f"  New storage range: [{storage_filtered.min():.1f}, {storage_filtered.max():.1f}] W/m²")
            
            if len(flux_filtered) > 10:  # Need minimum points for meaningful statistics
                # Calculate statistics on filtered data
                correlation = np.corrcoef(flux_filtered, storage_filtered)[0, 1]
                rmse = np.sqrt(np.mean((flux_filtered - storage_filtered)**2))
                mae = np.mean(np.abs(flux_filtered - storage_filtered))
                
                monthly_results[month_name] = {
                    'flux_values': flux_filtered.tolist(),
                    'storage_changes': storage_filtered.tolist(),
                    'correlation': correlation,
                    'rmse': rmse,
                    'mae': mae,
                    'num_segments': len(continuous_segments),
                    'total_days': len(month_data),
                    'num_points': len(flux_filtered),
                    'outliers_removed': len(flux_array) - len(flux_filtered)
                }
                
                print(f"  Correlation: {correlation:.3f}")
                print(f"  RMSE: {rmse:.1f} W/m²")
                print(f"  Data points: {len(flux_filtered)}")
            else:
                print(f"  Insufficient data points after outlier removal ({len(flux_filtered)} < 10)")
        else:
            print(f"  No valid data for {month_name}")
    
    return monthly_results

def segment_by_month(df, min_days, month_names):
    """Segment data by month with minimum day requirement"""
    segments = []
    for month_num, group in df.groupby('Month'):
        month_name = month_names[month_num]
        if len(group) >= min_days:
            print(f"{month_name}: {len(group)} days of data")
            segments.append((month_name, group))
        else:
            print(f"{month_name}: {len(group)} days of data (too few, skipping)")
    return segments

def generate_monthly_scatter_plots(monthly_results, save_path):
    """
    Generate scatter plots for each month with unified axis ranges and save to Google Drive
    
    Args:
        monthly_results: Dictionary containing monthly analysis results
        save_path: Path to save figures
    """
    # Define colors by season
    season_colors = {
        'Spring': '#66B2FF',  # Blue
        'Summer': '#FFCC99',  # Orange
        'Fall': '#99CC99'     # Green
    }
    
    month_season_map = {
        'Apr': 'Spring', 'May': 'Spring',
        'Jun': 'Summer', 'Jul': 'Summer', 'Aug': 'Summer',
        'Sep': 'Fall', 'Oct': 'Fall', 'Nov': 'Fall'
    }
    
    # Find global axis ranges
    all_flux_values = []
    all_storage_values = []
    
    for month_name, results in monthly_results.items():
        all_flux_values.extend(results['flux_values'])
        all_storage_values.extend(results['storage_changes'])
    
    if not all_flux_values or not all_storage_values:
        print("No data available for plotting")
        return
    
    global_min = min(min(all_flux_values), min(all_storage_values))
    global_max = max(max(all_flux_values), max(all_storage_values))
    
    # Add margin
    margin = (global_max - global_min) * 0.05
    axis_min = global_min - margin
    axis_max = global_max + margin
    
    print(f"\nGlobal axis range: [{axis_min:.1f}, {axis_max:.1f}] W/m²")
    
    # Generate plots for each month
    for month_name, results in monthly_results.items():
        print(f"\nGenerating scatter plot for {month_name}...")
        
        season = month_season_map.get(month_name, 'Summer')
        color = season_colors[season]
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(results['flux_values'], results['storage_changes'], 
                  alpha=0.6, color=color, s=20)
        
        # Set unified axis ranges
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        
        # Add diagonal line (perfect correlation)
        ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.5)
        
        # Set consistent ticks
        tick_spacing = (axis_max - axis_min) / 6
        ticks = np.arange(axis_min, axis_max + tick_spacing/2, tick_spacing)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        # Labels and formatting (no title)
        ax.set_xlabel('Energy Flux (W/m²)', fontsize=14, weight='bold')
        ax.set_ylabel('Energy Storage Change (W/m²)', fontsize=14, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Bold tick labels
        for tick in ax.get_xticklabels():
            tick.set_weight('bold')
        for tick in ax.get_yticklabels():
            tick.set_weight('bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{month_name}_EnergyFlux_vs_StorageChange.png"
        filepath = f"{save_path}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        
        plt.show()
    
    # Generate combined scatter plot with all months
    print(f"\nGenerating combined scatter plot with all months...")
    
    fig, ax = plt.subplots(figsize=(5, 5))  # Same size as individual plots
    
    # Plot data for each month with different colors
    for month_name, results in monthly_results.items():
        season = month_season_map.get(month_name, 'Summer')
        color = season_colors[season]
        ax.scatter(results['flux_values'], results['storage_changes'], 
                  alpha=0.4, color=color, s=20)  # Same point size as individual plots
    
    # Set unified axis ranges
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    # Add diagonal line (perfect correlation)
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.5)  # Same style as individual plots
    
    # Set consistent ticks
    tick_spacing = (axis_max - axis_min) / 6
    ticks = np.arange(axis_min, axis_max + tick_spacing/2, tick_spacing)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # Labels and formatting (same as individual plots)
    ax.set_xlabel('Energy Flux (W/m²)', fontsize=14, weight='bold')  # Same font size as individual plots
    ax.set_ylabel('Energy Storage Change (W/m²)', fontsize=14, weight='bold')  # Same font size as individual plots
    ax.tick_params(axis='both', which='major', labelsize=12)  # Same tick label size as individual plots
    
    # Bold tick labels
    for tick in ax.get_xticklabels():
        tick.set_weight('bold')
    for tick in ax.get_yticklabels():
        tick.set_weight('bold')
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Calculate overall statistics
    overall_correlation = np.corrcoef(all_flux_values, all_storage_values)[0, 1]
    overall_rmse = np.sqrt(np.mean((np.array(all_flux_values) - np.array(all_storage_values))**2))
    total_points = len(all_flux_values)
    
    print(f"  Overall correlation: {overall_correlation:.3f}")
    print(f"  Overall RMSE: {overall_rmse:.1f} W/m²")
    print(f"  Total data points: {total_points}")
    
    # Save combined figure to the same directory as individual plots
    filename = "All_Months_EnergyFlux_vs_StorageChange.png"
    filepath = f"{save_path}/{filename}"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    
    plt.show()


# Setup environment and get paths
print("Setting up environment...")
base_data_path = setup_environment()
file_paths = get_file_paths(base_data_path)

# Define save path for figures
base_save_path = "/content/drive/MyDrive/process_guided_deep_learning/Loss Function"

print(f"Using data path: {base_data_path}")
print(f"Using save path: {base_save_path}")

# Read the CSV files
print("Loading data...")
print("File paths:")
for key, path in file_paths.items():
    print(f"  {key}: {path}")

print("\nReading files...")
Lake_Mendota_Input = pd.read_csv(file_paths['weather'])
Lake_Mendota_Temperature = pd.read_csv(file_paths['temperature'])
Lake_Mendota_Bathymetry = pd.read_csv(file_paths['bathymetry'])
Lake_Mendota_Ice = pd.read_csv(file_paths['ice'])

print("Data files loaded successfully!")

# Data preprocessing
print("Preprocessing data...")
# Convert to datetime
print("  Converting dates...")
Lake_Mendota_Input['Date'] = pd.to_datetime(Lake_Mendota_Input['date'])
Lake_Mendota_Temperature['Date'] = pd.to_datetime(Lake_Mendota_Temperature['sampledate'])
Lake_Mendota_Ice['Date'] = pd.to_datetime(Lake_Mendota_Ice['date'])

# Merge datasets
print("  Merging datasets...")
Lake_Mendota = pd.merge(Lake_Mendota_Input, Lake_Mendota_Temperature, on='Date', how='inner')
Lake_Mendota = pd.merge(Lake_Mendota, Lake_Mendota_Ice[['Date', 'ice']], on='Date', how='inner')

# Filter out ice-covered periods
print("  Filtering ice-covered periods...")
Lake_Mendota = Lake_Mendota[Lake_Mendota['ice'] == False]

# Keep relevant columns
print("  Selecting relevant columns...")
Lake_Mendota = Lake_Mendota[['Date', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow', 'daynum', 'depth', 'wtemp']]

# Remove NaN values
print("  Removing NaN values...")
Lake_Mendota = Lake_Mendota.dropna()

# Filter for depths 0-20m
print("  Filtering for depths 0-20m...")
valid_depths = np.arange(0, 20.5, 1)
Lake_Mendota = Lake_Mendota[Lake_Mendota['depth'].isin(valid_depths)]

# Remove outliers using IQR method
print("  Removing outliers...")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Lake_Mendota = Lake_Mendota.groupby('depth').apply(lambda x: remove_outliers(x, 'wtemp')).reset_index(drop=True)

# Keep only dates with complete depth profiles (0-20m, 21 observations)
print("  Filtering for complete depth profiles...")
Lake_Mendota = Lake_Mendota.groupby('Date').filter(lambda x: (x['depth'].max() == 20) and (len(x) == 21))

print(f"Final dataset contains {Lake_Mendota['Date'].nunique()} unique dates")

# Prepare daily observations
weather_features = ['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow', 'daynum']
weather_day = Lake_Mendota.groupby('Date')[weather_features].mean()
temp_day = Lake_Mendota.groupby('Date')['wtemp'].apply(list).reset_index(name='Temperatures')
daily_observation = pd.merge(weather_day.reset_index(), temp_day, on='Date')

print(f"Daily observations: {len(daily_observation)} days")

# Process bathymetry data to get area array for 0-20m depths
print("Processing bathymetry data...")
depth = tf.constant(Lake_Mendota_Bathymetry["Depth(m)"], dtype=tf.float32)
area = tf.constant(Lake_Mendota_Bathymetry["Area(m^2)"], dtype=tf.float32)

# Linear regression to predict areas for integer depths
coeffs = tf_polyfit(depth, area, 1)
m, b = coeffs[0], coeffs[1]

# Calculate areas for depths 0-20m
desired_depth = tf.range(0, 21, delta=1, dtype=tf.float32)
predicted_area = m * desired_depth + b
area_array = predicted_area

print(f"Area array shape: {area_array.shape}")

# Filter data for months 4-11 (April-November)
daily_observation['Month'] = daily_observation['Date'].dt.month
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

print("\nData count by month:")
month_counts = daily_observation['Month'].value_counts().sort_index()
for month, count in month_counts.items():
    print(f"{month_names[month]} (Month {month}): {count} days")

# Filter for months 4-11
target_months = [4, 5, 6, 7, 8, 9, 10, 11]
filtered_observation = daily_observation[daily_observation['Month'].isin(target_months)]
print(f"\nFiltered data (Apr-Nov): {len(filtered_observation)} days")

# Execute the analysis
print("\n" + "="*60)
print("MONTHLY ENERGY BALANCE ANALYSIS (FULL DEPTH 0-20m)")
print("="*60)

# Segment data by month
monthly_segments = segment_by_month(filtered_observation, min_days=5, month_names=month_names)

# Analyze energy balance for each month
monthly_results = analyze_monthly_energy_balance(monthly_segments, area_array)

# Print summary
print("\n" + "="*60)
print("SUMMARY OF ENERGY BALANCE ANALYSIS")
print("="*60)
for month_name, results in monthly_results.items():
    print(f"{month_name:8s}: r={results['correlation']:6.3f}, RMSE={results['rmse']:6.1f} W/m², "
          f"Points={results['num_points']:4d}, Removed={results['outliers_removed']:3d}")

# Generate scatter plots
print("\n" + "="*60)
print("GENERATING SCATTER PLOTS")
print("="*60)

# Create save directory for figures
os.makedirs(base_save_path, exist_ok=True)
print(f"Figures will be saved to: {base_save_path}")

generate_monthly_scatter_plots(monthly_results, base_save_path)

print("\nAnalysis completed!")
print("All calculations use full depth integration (0-20m)")
print("Outlier threshold: ±500 W/m²")