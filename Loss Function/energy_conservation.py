import torch
import torch.nn as nn

class EnergyConservation:
    """Handles all energy conservation calculations using full depth (0-20m)"""
    def __init__(self, device=None):
        # Automatically detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"  # GPU on Colab
            elif torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU on Mac
            else:
                device = "cpu"  # Default to CPU

        self.device = device
        print(f"Using device: {self.device}")
        
        # Get area array from DataProcessor
        from data_preprocessing import DataProcessor
        processor = DataProcessor()
        self.area_array = torch.tensor(processor.area_array, dtype=torch.float32).to(self.device)
        
        # Initialize all constants
        self.constants = {
            # Vapor pressure constants
            'rh_scaling': torch.tensor(1.0, dtype=torch.float32).to(device),
            
            # Air density constants
            'mwrw2a': torch.tensor(18.016 / 28.966, dtype=torch.float32).to(device),
            'c_gas': torch.tensor(1.0e3 * 8.31436 / 28.966, dtype=torch.float32).to(device),
            'p': torch.tensor(1013.0, dtype=torch.float32).to(device),
            
            # Wind speed constants
            'c_z0': torch.tensor(0.001, dtype=torch.float32).to(device),
            'height_10m': torch.tensor(10.0, dtype=torch.float32).to(device),
            'ref_height': torch.tensor(2.0, dtype=torch.float32).to(device),
            
            # Heat flux constants
            'c_E': torch.tensor(0.0013, dtype=torch.float32).to(device),
            'lambda_v': torch.tensor(2.453e6, dtype=torch.float32).to(device),
            'omega': torch.tensor(0.622, dtype=torch.float32).to(device),
            'c_a': torch.tensor(1005.0, dtype=torch.float32).to(device),
            'c_H': torch.tensor(0.0013, dtype=torch.float32).to(device),
            
            # Energy flux constants
            'e_s': torch.tensor(0.985, dtype=torch.float32).to(device),
            'alpha_sw': torch.tensor(0.07, dtype=torch.float32).to(device),
            'alpha_lw': torch.tensor(0.03, dtype=torch.float32).to(device),
            'sigma': torch.tensor(5.67e-8, dtype=torch.float32).to(device),
            
            # Lake energy constant
            'c_w': torch.tensor(4186.0, dtype=torch.float32).to(device)  # Specific heat of water
        }

    def calculate_vapour_pressure_saturated(self, temp):
        """Calculate saturated vapor pressure."""
        exponent = (9.28603523 - (2332.37885/(temp+273.15))) * torch.log(torch.tensor(10.0).to(self.device))
        return torch.exp(exponent)

    def calculate_vapour_pressure_air(self, rel_hum, temp):
        """Calculate vapor pressure."""
        return (self.constants['rh_scaling'] * (rel_hum / 100.0) * 
                self.calculate_vapour_pressure_saturated(temp))

    def calculate_air_density(self, air_temp, rh):
        """Calculate air density."""
        vap_pressure = self.calculate_vapour_pressure_air(rh, air_temp)
        r = self.constants['mwrw2a'] * vap_pressure / (self.constants['p'] - vap_pressure)
        return ((1.0 / self.constants['c_gas'] * (1 + r) / (1 + r / self.constants['mwrw2a']) * 
                self.constants['p'] / (air_temp + 273.15)) * 100)

    def calculate_wind_speed_10m(self, ws):
        """Calculate wind speed at 10m."""
        return ws * (torch.log(self.constants['height_10m'] / self.constants['c_z0']) /
                    torch.log(self.constants['ref_height'] / self.constants['c_z0']))

    def calculate_heat_flux_latent(self, surf_temp, air_temp, rel_hum, wind_speed):
        """
        Calculate latent heat flux using surface temperature.
        All inputs should be shape [batch_size]
        """
        rho_a = self.calculate_air_density(air_temp, rel_hum)
        U_10 = self.calculate_wind_speed_10m(wind_speed)
        
        e_s = self.calculate_vapour_pressure_saturated(surf_temp)
        e_a = self.calculate_vapour_pressure_air(rel_hum, air_temp)
        
        return (-rho_a * self.constants['c_E'] * self.constants['lambda_v'] * U_10 * 
                (self.constants['omega'] / self.constants['p']) * (e_s - e_a))

    def calculate_heat_flux_sensible(self, surf_temp, air_temp, rel_hum, wind_speed):
        """
        Calculate sensible heat flux using surface temperature.
        All inputs should be shape [batch_size]
        """
        rho_a = self.calculate_air_density(air_temp, rel_hum)
        U_10 = self.calculate_wind_speed_10m(wind_speed)
        
        return -rho_a * self.constants['c_a'] * self.constants['c_H'] * U_10 * (surf_temp - air_temp)

    def calculate_energy_fluxes(self, shortwave_prev, longwave_prev, air_temp_prev, rel_hum_prev, wind_speed_prev,
                              shortwave_curr, longwave_curr, air_temp_curr, rel_hum_curr, wind_speed_curr,
                              surf_temp_prev, surf_temp_curr):
        """
        Calculate all energy fluxes using surface temperatures.
        All inputs should be shape [batch_size]
        Returns flux with shape [batch_size]
        """
        # Calculate outgoing longwave radiation using surface temperature
        R_lw_out_prev = self.constants['e_s'] * self.constants['sigma'] * torch.pow(surf_temp_prev + 273.15, 4)
        R_lw_out_curr = self.constants['e_s'] * self.constants['sigma'] * torch.pow(surf_temp_curr + 273.15, 4)

        # Calculate mean values
        R_sw_mean = 0.5 * (shortwave_prev + shortwave_curr)
        R_lw_mean = 0.5 * (longwave_prev + longwave_curr)
        R_lw_out_mean = 0.5 * (R_lw_out_prev + R_lw_out_curr)

        # Calculate latent heat flux
        E = 0.5 * (
            self.calculate_heat_flux_latent(surf_temp_prev, air_temp_prev, rel_hum_prev, wind_speed_prev) +
            self.calculate_heat_flux_latent(surf_temp_curr, air_temp_curr, rel_hum_curr, wind_speed_curr)
        )

        # Calculate sensible heat flux
        H = 0.5 * (
            self.calculate_heat_flux_sensible(surf_temp_prev, air_temp_prev, rel_hum_prev, wind_speed_prev) +
            self.calculate_heat_flux_sensible(surf_temp_curr, air_temp_curr, rel_hum_curr, wind_speed_curr)
        )

        return (R_sw_mean * (1 - self.constants['alpha_sw']) +
                R_lw_mean * (1 - self.constants['alpha_lw']) -
                R_lw_out_mean + E + H)

    def calculate_density(self, temp):
        """Calculate water density."""
        return 1000 * (1 - (temp + 288.9414) * (temp - 3.9863)**2 / (508929.2 * (temp + 68.12963)))
    
    def calculate_full_depth_lake_energy(self, temps, densities, unit_depth=1.0):
        """
        Calculate lake energy using full depth (0-20m)
        
        Args:
            temps: Temperature tensor [batch_size, n_depths]
            densities: Density tensor [batch_size, n_depths]
            unit_depth: Unit depth value
            
        Returns:
            Energy tensor [batch_size]
        """
        # Calculate energy for all 21 depths
        energy = torch.sum(
            temps * densities * self.area_array * unit_depth * self.constants['c_w'],
            dim=-1
        )
        return energy