# Set the working directory to the Simulation folder before running this script
# e.g. setwd("path/to/process_guided_deep_learning/Simulation")
sim_folder <- getwd()

# Load library
library(glmtools)
library(GLM3r)
library(rLakeAnalyzer)
library(tidyverse)
library(ncdf4)

# Read data
glm_template = 'glm3-template.nml' 
output_file <- file.path(sim_folder, "output","output.nc")
field_data <- file.path(sim_folder,"bcs","field_temp_oxy.csv")
file.copy(glm_template, 'glm3.nml', overwrite = TRUE)
nml_file <- file.path(sim_folder, 'glm3.nml')

# Set the calibrated parameters
calibrated_params <- data.frame(pars = c('wind_factor', 'lw_factor', 'ch', 'coef_mix_hyp', 'Kw'), calibrated = c(0.9217, 0.9461, 0.0015, 0.5380, 0.2594))

# Read the current name list file
nml <- read_nml(nml_file)

# Update parameters
for (i in 1:nrow(calibrated_params)) {
  nml <- set_nml(nml, calibrated_params$pars[i], calibrated_params$calibrated[i])
}

# Write the updated name list file
write_nml(nml, nml_file)

# Run the model with updated parameters
GLM3r::run_glm(sim_folder, verbose = TRUE)

# Read the output
nc_file <- file.path(sim_folder, 'output', 'output.nc')

# Calculate RMSE with the new parameters
calibrated_temp_rmse <- compare_to_field(nc_file = nc_file, field_file = field_data, metric = 'water.temperature', as_value = FALSE, precision = 'hours')
print(paste('Total time period (calibrated):', round(calibrated_temp_rmse, 2), 'deg C RMSE'))

# Read the output data
nc_file <- file.path(sim_folder, 'output', 'output.nc')
nc <- nc_open(nc_file)

# Get the time data
time <- ncvar_get(nc, "time")
time_units <- ncatt_get(nc, "time", "units")$value

# Manually set the origin date
origin_date <- ymd_hms("2009-01-01 12:00:00")

# Convert time to dates
dates <- as.Date(as.POSIXct(time * 3600, origin = "2009-01-01", tz = "UTC"))

# Get depth data
depth <- ncvar_get(nc, "z")

# Get temperature data
temp <- ncvar_get(nc, "temp")

# Close the output file
nc_close(nc)

# Initialize an empty list to store the processed data
processed_data <- list()

# Process the depth and temperature data
for (i in 1:ncol(depth)) {
  
  # Extract the daily depth and temperature
  daily_depth <- depth[,i]
  daily_temp <- temp[,i]
  
  # Remove depths > 25 and corresponding temperatures
  valid_indices <- daily_depth <= 25 & !is.na(daily_temp)
  valid_depth <- daily_depth[valid_indices]
  valid_temp <- daily_temp[valid_indices]
  
  # Reverse depth to match temperature
  valid_depth <- rev(valid_depth)
  
  # Create a data frame for this day
  daily_data <- data.frame(
    date = dates[i],
    depth = valid_depth,
    temperature = valid_temp
  )
  
  # Add to the list
  processed_data[[i]] <- daily_data
}

# Combine all processed data
simulate_temp <- do.call(rbind, processed_data)

# Round depth to the nearest meter
simulate_temp$depth <- round(simulate_temp$depth)

# Group by date and depth, then calculate the mean temperature
simulate_temp <- simulate_temp %>% group_by(date, depth) %>% summarise(temperature = mean(temperature, na.rm = TRUE)) %>% ungroup()

# Output the simulation data
write.csv(simulate_temp, "simulate_temp.csv", row.names = FALSE)




