# Import libraries
import torch
import torch.nn as nn
from energy_conservation import EnergyConservation

class EnsembleEnergyConservation:
    """Integrates energy conservation with ensemble model predictions using full depth (0-20m)"""
    def __init__(self, device=None):
        # Create energy conservation calculator
        self.energy_calculator = EnergyConservation(device)
        self.device = self.energy_calculator.device
        
        # Get surface area for unit conversion
        self.surface_area = self.energy_calculator.area_array[0]  # Surface area (first element)
        
    def calculate_energy_loss(self, pred_temps, temp_day29, weather_day29_30, daynum, threshold=0):
        """
        Calculate energy conservation loss using full depth (0-20m)
        
        Args:
            pred_temps: Predicted temperatures for day 30 [batch_size, n_depths]
            temp_day29: Day 29 temperatures [batch_size, n_depths]
            weather_day29_30: Day 29-30 weather [batch_size, 2, n_features] 
            daynum: Day of year [batch_size] - NOT USED (kept for compatibility)
            threshold: Energy difference threshold (only penalize differences above this threshold)
            
        Returns:
            Energy conservation loss
        """
        # Extract weather data for day 29 and day 30
        weather_29 = weather_day29_30[:, 0, :]  # Day 29 weather
        weather_30 = weather_day29_30[:, 1, :]  # Day 30 weather
        
        # Calculate energy storage using FULL DEPTH
        storage_29 = self._calculate_full_depth_energy_storage(temp_day29)
        storage_30 = self._calculate_full_depth_energy_storage(pred_temps)
        
        # Calculate storage change (Day 29 → Day 30) in Joules
        storage_change = storage_30 - storage_29
        
        # Convert storage change from Joules to W/m² (divide by seconds per day and surface area)
        storage_change_wm2 = storage_change / (86400 * self.surface_area)  # J to W/m²
        
        # Calculate energy flux for the period (29 → 30) - already in W/m²
        energy_flux = self._calculate_flux(weather_29, weather_30, temp_day29, pred_temps)
        
        # Energy conservation constraint: storage change should equal flux
        energy_diff = torch.abs(storage_change_wm2 - energy_flux)
        
        # Apply threshold - only penalize differences exceeding threshold
        threshold_tensor = torch.tensor(threshold, dtype=torch.float32, device=self.device)
        thresholded_diff = torch.clamp(energy_diff - threshold_tensor, min=0)
        
        # Compute average over all samples
        loss = torch.mean(thresholded_diff)
        
        return loss

    def _calculate_full_depth_energy_storage(self, temperatures):
        """
        Calculate energy storage using full depth (0-20m)
        
        Args:
            temperatures: Temperature profile [batch_size, n_depths]
            
        Returns:
            Total energy storage [batch_size] in Joules
        """
        # Calculate densities
        densities = self.energy_calculator.calculate_density(temperatures)
        
        # Calculate lake energy using full depth
        return self.energy_calculator.calculate_full_depth_lake_energy(
            temperatures, densities, unit_depth=1.0
        )
    
    def _calculate_flux(self, weather_prev, weather_curr, temp_prev, temp_curr):
        """
        Calculate energy flux between two time periods
        
        Args:
            weather_prev: Weather at previous time step [batch_size, n_features]
            weather_curr: Weather at current time step [batch_size, n_features]
            temp_prev: Temperatures at previous time step [batch_size, n_depths]
            temp_curr: Temperatures at current time step [batch_size, n_depths]
            
        Returns:
            Energy flux [batch_size] in W/m²
        """
        # Extract weather variables for previous time step
        shortwave_prev = weather_prev[..., 0]
        longwave_prev = weather_prev[..., 1]
        air_temp_prev = weather_prev[..., 2]
        rel_hum_prev = weather_prev[..., 3]
        wind_speed_prev = weather_prev[..., 4]
        
        # Extract weather variables for current time step
        shortwave_curr = weather_curr[..., 0]
        longwave_curr = weather_curr[..., 1]
        air_temp_curr = weather_curr[..., 2]
        rel_hum_curr = weather_curr[..., 3]
        wind_speed_curr = weather_curr[..., 4]
        
        # Get surface temperatures (first depth)
        surf_temp_prev = temp_prev[..., 0]
        surf_temp_curr = temp_curr[..., 0]
        
        # Use existing calculate_energy_fluxes method (returns W/m²)
        return self.energy_calculator.calculate_energy_fluxes(
            shortwave_prev, longwave_prev, air_temp_prev, rel_hum_prev, wind_speed_prev,
            shortwave_curr, longwave_curr, air_temp_curr, rel_hum_curr, wind_speed_curr,
            surf_temp_prev, surf_temp_curr
        )
    
    def get_weighted_loss(self, mse_loss, energy_loss, energy_weight):
        """
        Calculate weighted loss combining MSE and energy conservation
        
        Args:
            mse_loss: Mean squared error loss for temperature predictions
            energy_loss: Energy conservation loss
            energy_weight: Weight for energy conservation loss
            
        Returns:
            Combined loss
        """
        return mse_loss + energy_weight * energy_loss