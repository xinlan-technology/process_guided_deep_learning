# Import libraries
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random, os, json, logging, copy
from datetime import datetime

def set_seed(seed=42):
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
        self.max_evals = 400
        self.epoch_number = 400
        self.early_stopping_patience = 30
        self.batch_size = 32
        self.results_dir = os.path.join(os.getcwd(), 'optimization_results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Hidden size options (shared across LSTM variants)
        hidden_size_choices = [64, 96, 128, 160, 192, 256]

        # LSTM and Attention-LSTM: simplified structure with layer-wise decay
        self.lstm_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
            'hidden_size': hp.choice('hidden_size', hidden_size_choices)
        }

        self.attention_lstm_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
            'hidden_size': hp.choice('hidden_size', hidden_size_choices)
        }

        # CNN_LSTM: restrict filter sizes and kernel width to control complexity
        self.cnn_lstm_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
            'hidden_size': hp.choice('hidden_size', hidden_size_choices),
            'num_filters_l1': hp.choice('num_filters_l1', [40, 60, 80]),
            'num_filters_l2': hp.choice('num_filters_l2', [20, 35, 50]),
            'num_filters_l3': hp.choice('num_filters_l3', [5, 10, 15]),
            'kernel_size': hp.choice('kernel_size', [3, 5, 7])
        }

        # Define compatible (d_model, nhead) pairs to avoid assertion errors in PyTorch Transformers
        valid_transformer_pairs = [
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

        # Transformer: restrict dropout, enforce compatible (d_model, nhead), and control feedforward expansion
        self.transformer_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
            'transformer_config': hp.choice('transformer_config', valid_transformer_pairs),
            'num_layers': hp.choice('num_layers', [2, 3, 4]),
            'dim_feedforward_factor': hp.choice('dim_feedforward_factor', [2, 4, 6])
        }

class ModelTrainer:
    def __init__(self, model_class, train_loader, valid_loader, input_size, output_size, **kwargs):
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
        logfile = os.path.join(self.config.results_dir, f'opt_{self.model_class_name}_{ts}.log')
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(logfile), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

    def get_parameter_space(self):
        if self.model_class_name == 'TransformerModel':
            return self.config.transformer_space
        elif self.model_class_name == 'CNN_LSTM':
            return self.config.cnn_lstm_space
        elif self.model_class_name == 'AttentionLSTM':
            return self.config.attention_lstm_space
        else:
            return self.config.lstm_space

    def objective(self, params):
        try:
            self.logger.info(f"Trying: {params}")

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

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.epoch_number):
                model.train()
                for batch in self.train_loader:
                    X, y = batch[0].to(self.device), batch[1].to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(X), y)
                    loss.backward()
                    optimizer.step()

                val_loss = self.validate(model, criterion)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    break

            if best_val_loss < self.best_trial['loss']:
                self.best_trial.update({
                    'loss': best_val_loss,
                    'params': copy.deepcopy(params),
                    'model_state': best_state,
                    'status': STATUS_OK
                })
                self.save_best_trial()

            return {'loss': best_val_loss, 'status': STATUS_OK}

        except Exception as e:
            self.logger.error(f"Trial failed: {e}")
            return {'loss': float('inf'), 'status': 'fail'}

    def validate(self, model, criterion):
        model.eval()
        loss, count = 0.0, 0
        with torch.no_grad():
            for batch in self.valid_loader:
                X, y = batch[0].to(self.device), batch[1].to(self.device)
                loss += criterion(model(X), y).item()
                count += 1
        return loss / max(1, count)

    def save_best_trial(self):
        model_path = os.path.join(self.config.results_dir, f'best_model_{self.model_class_name}.pth')
        torch.save(self.best_trial['model_state'], model_path)
        self.logger.info(f"Saved best model to {model_path}")

        params_path = os.path.join(self.config.results_dir, f'best_params_{self.model_class_name}.json')
        with open(params_path, 'w') as f:
            json.dump({k: (v.item() if hasattr(v, 'item') else v) for k, v in self.best_trial['params'].items()}, f, indent=4)
        self.logger.info(f"Saved best parameters to {params_path}")

    def get_actual_params(self, params):
        return space_eval(self.get_parameter_space(), params)

    def optimize(self):
        self.logger.info(f"Starting hyperparameter optimization for {self.model_class_name}")
        self.trials = Trials()
        best = fmin(
            fn=self.objective,
            space=self.get_parameter_space(),
            algo=tpe.suggest,
            max_evals=self.config.max_evals,
            trials=self.trials,
            rstate=np.random.default_rng(42)
        )
        if self.best_trial['loss'] != float('inf'):
            self.logger.info(f"Optimization complete. Best validation loss: {self.best_trial['loss']:.6f}")
            print(f"FINAL BEST VALIDATION LOSS: {self.best_trial['loss']:.6f}")
        else:
            self.logger.warning("Optimization did not find a valid solution.")
            print("WARNING: Optimization did not find a valid solution.")
        return self.best_trial

def run_optimization(model_class, train_loader, valid_loader, input_size, output_size):
    trainer = ModelTrainer(model_class, train_loader, valid_loader, input_size, output_size)
    return trainer.optimize()
