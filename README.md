# Process-guided deep learning for lake water temperature

This repository contains the code used to predict daily water temperature profiles (0-20 m) in Lake Mendota using deep learning models with physical constraints.

## Overview

The project investigates four aspects of lake temperature modeling:

1. **Base Models** — Compare four deep learning architectures (`LSTM`, `Transformer`, `CNN-LSTM`, `AttentionLSTM`) trained on observational data with varying training set sizes (20%-100%).
2. **Pretraining** — Pretrain models on simulation data from the General Lake Model (GLM), then finetune on real observations.
3. **Ensemble** — Combine predictions from the four base models using a depth-wise weighted ensemble.
4. **Process-Guided Loss** — Add an energy conservation constraint to the loss function to improve physical consistency.

## Repository Structure

```text
process_guided_deep_learning/
├── Base Model/              # Train four DL models on observational data
├── Pretraining/             # Pretrain on simulation, then finetune on observations
├── Ensemble/                # Depth-wise weighted ensemble of base models
├── Loss Function/           # Energy-conservation-constrained ensemble
├── Simulation/              # GLM setup and simulation script
└── Validation/              # Monthly energy balance analysis
```

## Environment and Data Setup

This repository was developed primarily for **Google Colab + Google Drive**.

The project uses daily weather and water temperature data from Lake Mendota, together with lake bathymetry, ice-cover records, and GLM simulation output. The expected input files are referenced in each module's `environment_configuration.py`.

Before running the scripts:

1. Mount Google Drive in Colab.
2. Update the paths in each `environment_configuration.py` file if your folder structure is different.
3. Check the additional hard-coded Google Drive paths in the main training and evaluation scripts, such as:
   - `Base Model/basemodel_training.py`
   - `Base Model/figures_plot.py`
   - `Pretraining/basemodel_pretraining.py`
   - `Pretraining/basemodel_training.py`
   - `Pretraining/figures_plot.py`
   - `Ensemble/ensemble_data_processing.py`
   - `Ensemble/ensemble_evaluation.py`
   - `Ensemble/parameter_tuning_ensemble.py`
   - `Loss Function/ensemble_data_processing.py`
   - `Loss Function/energy_parameter_tuning.py`
   - `Loss Function/ensemble_energy_evaluation.py`
   - `Validation/monthly_energy_analysis.py`

## Requirements

- Python 3.8+
- PyTorch
- TensorFlow
- hyperopt
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- tqdm
- For simulation: R with packages `GLM3r`, `glmtools`, `rLakeAnalyzer`, `tidyverse`, `ncdf4`

## Recommended Execution Order

### 1. Simulation

If you want to regenerate GLM simulation output:

- Run `Simulation/general_lake_model.R`
- This script should be launched from the `Simulation` folder

### 2. Base Model

- Run `Base Model/basemodel_training.py`
- Run `Base Model/figures_plot.py`

### 3. Pretraining

- Run `Pretraining/basemodel_pretraining.py`
- Run `Pretraining/basemodel_training.py`
- Run `Pretraining/figures_plot.py`

### 4. Ensemble

- Run `Ensemble/parameter_tuning_ensemble.py`
- Run `Ensemble/ensemble_evaluation.py`

### 5. Process-Guided Loss

- Run `Loss Function/energy_parameter_tuning.py`
- Run `Loss Function/ensemble_energy_evaluation.py`

### 6. Validation

- Run `Validation/monthly_energy_analysis.py`

## Notes

- The repository is organized around comparative experiments on model architecture, pretraining, ensemble learning, and process-guided loss design.
- The current implementation uses random splitting of sliding-window samples, so the code is most appropriate for comparative evaluation under a shared protocol.
