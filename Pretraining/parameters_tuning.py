# Import libraries
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random, os, json, logging, copy
from datetime import datetime

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Pretraining optimization parameters
MAX_EVALS = 400                    # Number of hyperparameter evaluations (same as original)
MAX_EPOCHS = 400                   # Maximum training epochs per evaluation (same as original)
EARLY_STOPPING_PATIENCE = 30      # Early stopping patience (same as original)
BATCH_SIZE = 32                    # Training batch size

# Optimizer parameters
WEIGHT_DECAY = 1e-5               # L2 regularization
SCHEDULER_FACTOR = 0.5            # Learning rate reduction factor
SCHEDULER_PATIENCE = 15           # Scheduler patience
MIN_LEARNING_RATE = 1e-6          # Minimum learning rate

# Hidden size options (shared across LSTM variants)
HIDDEN_SIZE_CHOICES = [64, 96, 128, 160, 192, 256]

# CNN filter options
CNN_FILTERS_L1 = [40, 60, 80]
CNN_FILTERS_L2 = [20, 35, 50]
CNN_FILTERS_L3 = [5, 10, 15]
CNN_KERNEL_SIZES = [3, 5, 7]

# Transformer configuration pairs (d_model, nhead)
TRANSFORMER_CONFIGS = [
    {'d_model': 128, 'nhead': 2},
    {'d_model': 128, 'nhead': 4},
    {'d_model': 128, 'nhead': 8},
    {'d_model': 192, 'nhead': 2},
    {'d_model': 192, 'nhead': 4},
    {'d_model': 192, 'nhead': 8},
    {'d_model': 256, 'nhead': 2},
    {'d_model': 256, 'nhead': 4},
    {'d_model': 256, 'nhead': 8}
]

# Transformer options
TRANSFORMER_NUM_LAYERS = [2, 3, 4]
TRANSFORMER_FF_FACTORS = [2, 4, 6]

# =============================================================================
# MAIN CLASSES
# =============================================================================

def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class ModelConfig:
    def __init__(self):
        # Use configuration parameters
        self.max_evals = MAX_EVALS
        self.epoch_number = MAX_EPOCHS
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        self.batch_size = BATCH_SIZE
        self.results_dir = os.path.join(os.getcwd(), 'optimization_results')
        os.makedirs(self.results_dir, exist_ok=True)

        # LSTM and Attention-LSTM hyperparameter spaces
        self.lstm_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
            'hidden_size': hp.choice('hidden_size', HIDDEN_SIZE_CHOICES)
        }

        self.attention_lstm_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
            'hidden_size': hp.choice('hidden_size', HIDDEN_SIZE_CHOICES)
        }

        # CNN_LSTM hyperparameter space
        self.cnn_lstm_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
            'hidden_size': hp.choice('hidden_size', HIDDEN_SIZE_CHOICES),
            'num_filters_l1': hp.choice('num_filters_l1', CNN_FILTERS_L1),
            'num_filters_l2': hp.choice('num_filters_l2', CNN_FILTERS_L2),
            'num_filters_l3': hp.choice('num_filters_l3', CNN_FILTERS_L3),
            'kernel_size': hp.choice('kernel_size', CNN_KERNEL_SIZES)
        }

        # Transformer hyperparameter space
        self.transformer_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
            'transformer_config': hp.choice('transformer_config', TRANSFORMER_CONFIGS),
            'num_layers': hp.choice('num_layers', TRANSFORMER_NUM_LAYERS),
            'dim_feedforward_factor': hp.choice('dim_feedforward_factor', TRANSFORMER_FF_FACTORS)
        }

class ModelTrainer:
    def __init__(self, model_class, train_loader, valid_loader, input_size, output_size):
        """
        Initialize trainer for pretraining phase only.
        Finetuning is now handled separately with fixed hyperparameters.
        """
        set_seed()
        self.config = ModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = model_class
        self.model_class_name = model_class.__name__
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.input_size = input_size
        self.output_size = output_size
        self.setup_logging()
        self.best_trial = {'loss': float('inf'), 'params': None, 'model_state': None, 'status': None}
        self.trials = None

    def setup_logging(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        logfile = os.path.join(self.config.results_dir, f'pretrain_{self.model_class_name}_{ts}.log')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(logfile), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

    def get_parameter_space(self):
        """Get hyperparameter search space for the model"""
        if self.model_class_name == 'TransformerModel':
            return self.config.transformer_space
        elif self.model_class_name == 'CNN_LSTM':
            return self.config.cnn_lstm_space
        elif self.model_class_name == 'AttentionLSTM':
            return self.config.attention_lstm_space
        else:
            return self.config.lstm_space

    def objective(self, params):
        """Objective function for hyperparameter optimization"""
        try:
            self.logger.info(f"Trying parameters: {params}")

            # Create model instance with proper parameter unpacking
            if self.model_class_name == 'TransformerModel':
                cfg = params['transformer_config']
                model = self.model_class(
                    input_size=self.input_size,
                    d_model=cfg['d_model'],
                    nhead=cfg['nhead'],
                    output_size=self.output_size,
                    dropout_rate=params['dropout_rate'],
                    num_layers=params['num_layers'],
                    dim_feedforward_factor=params['dim_feedforward_factor']
                ).to(self.device)

            elif self.model_class_name == 'CNN_LSTM':
                model = self.model_class(
                    input_size=self.input_size,
                    output_size=self.output_size,
                    dropout_rate=params['dropout_rate'],
                    hidden_size1=params['hidden_size'],
                    hidden_size2=params['hidden_size'] // 2,
                    hidden_size3=params['hidden_size'] // 4,
                    num_filters_l1=params['num_filters_l1'],
                    num_filters_l2=params['num_filters_l2'],
                    num_filters_l3=params['num_filters_l3'],
                    kernel_size=params['kernel_size']
                ).to(self.device)

            elif self.model_class_name in ['LSTMModel', 'AttentionLSTM']:
                model = self.model_class(
                    input_size=self.input_size,
                    output_size=self.output_size,
                    dropout_rate=params['dropout_rate'],
                    hidden_size1=params['hidden_size'],
                    hidden_size2=params['hidden_size'] // 2,
                    hidden_size3=params['hidden_size'] // 4
                ).to(self.device)

            else:
                raise ValueError(f"Unsupported model class: {self.model_class_name}")

            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=SCHEDULER_FACTOR, 
                patience=SCHEDULER_PATIENCE, min_lr=MIN_LEARNING_RATE
            )
            
            best_val_loss = float('inf')
            best_state = None
            patience_counter = 0

            # Training loop
            for epoch in range(self.config.epoch_number):
                # Training phase
                model.train()
                for batch in self.train_loader:
                    X, y = batch[0].to(self.device), batch[1].to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(X), y)
                    loss.backward()
                    optimizer.step()

                # Validation phase
                val_loss = self.validate(model, criterion)
                scheduler.step(val_loss)

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Update best trial if this is better
            if best_val_loss < self.best_trial['loss']:
                self.best_trial.update({
                    'loss': best_val_loss,
                    'params': copy.deepcopy(params),
                    'model_state': best_state,
                    'status': STATUS_OK
                })
                self.save_best_trial()

            self.logger.info(f"Trial completed. Validation loss: {best_val_loss:.6f}")
            return {'loss': best_val_loss, 'status': STATUS_OK}

        except Exception as e:
            self.logger.error(f"Trial failed with error: {e}")
            return {'loss': float('inf'), 'status': 'fail'}

    def validate(self, model, criterion):
        """Validate model on validation set"""
        model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in self.valid_loader:
                X, y = batch[0].to(self.device), batch[1].to(self.device)
                loss = criterion(model(X), y)
                total_loss += loss.item()
                count += 1
                
        return total_loss / max(1, count)

    def save_best_trial(self):
        """Save the best model and parameters found so far"""
        # Save model
        model_path = os.path.join(self.config.results_dir, f'best_pretrain_{self.model_class_name}.pth')
        torch.save(self.best_trial['model_state'], model_path)
        self.logger.info(f"Saved best pretrained model to {model_path}")

        # Save parameters
        params_path = os.path.join(self.config.results_dir, f'best_pretrain_params_{self.model_class_name}.json')
        params_to_save = {}
        for k, v in self.best_trial['params'].items():
            if hasattr(v, 'item'):
                params_to_save[k] = v.item()
            else:
                params_to_save[k] = v
                
        with open(params_path, 'w') as f:
            json.dump(params_to_save, f, indent=4)
        self.logger.info(f"Saved best parameters to {params_path}")

    def optimize(self):
        """Run hyperparameter optimization for pretraining"""
        self.logger.info(f"Starting pretraining hyperparameter optimization for {self.model_class_name}")
        self.logger.info(f"Search space: {self.get_parameter_space()}")
        
        self.trials = Trials()
        
        try:
            best = fmin(
                fn=self.objective,
                space=self.get_parameter_space(),
                algo=tpe.suggest,
                max_evals=self.config.max_evals,
                trials=self.trials,
                rstate=np.random.default_rng(42)
            )
            
            if self.best_trial['loss'] != float('inf'):
                self.logger.info(f"Pretraining optimization completed successfully!")
                self.logger.info(f"Best validation loss: {self.best_trial['loss']:.6f}")
                self.logger.info(f"Best parameters: {self.best_trial['params']}")
                print(f"PRETRAINING COMPLETED - Best validation loss: {self.best_trial['loss']:.6f}")
            else:
                self.logger.warning("Pretraining optimization failed to find valid solution.")
                print("WARNING: Pretraining optimization failed.")
                
        except Exception as e:
            self.logger.error(f"Optimization failed with error: {e}")
            print(f"ERROR: Optimization failed - {e}")
            
        return self.best_trial

def run_optimization(model_class, train_loader, valid_loader, input_size, output_size):
    """
    Run hyperparameter optimization for pretraining phase.
    
    This function is now only used for pretraining. Finetuning uses fixed
    hyperparameters with a custom training loop.
    """
    trainer = ModelTrainer(model_class, train_loader, valid_loader, input_size, output_size)
    return trainer.optimize()