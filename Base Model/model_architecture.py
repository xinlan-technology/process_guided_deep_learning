# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTMModel(nn.Module):
    """LSTM model implementation"""
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, 
                 output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        
        # First LSTM layer and its auxiliary layers
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second LSTM layer and its auxiliary layers
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third LSTM layer and its auxiliary layers
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size3, hidden_size3 // 2)
        self.fc2 = nn.Linear(hidden_size3 // 2, output_size)
    
    def _process_lstm_layer(self, x, lstm_layer, bn_layer, dropout_layer):
        """Helper method to process data through one LSTM layer"""
        x, _ = lstm_layer(x)
        x = x.permute(0, 2, 1)
        x = bn_layer(x)
        x = x.permute(0, 2, 1)
        x = dropout_layer(x)
        return x
        
    def forward(self, x):
        # Process through LSTM layers
        x = self._process_lstm_layer(x, self.lstm1, self.bn1, self.dropout1)
        x = self._process_lstm_layer(x, self.lstm2, self.bn2, self.dropout2)
        x = self._process_lstm_layer(x, self.lstm3, self.bn3, self.dropout3)
        
        # Select the output of the last time step
        x = x[:, -1, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding using sinusoidal functions.

    This module injects positional information into the input embeddings 
    to give the model awareness of sequence order.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a [max_len, d_model] matrix for positional encodings
        pe = torch.zeros(max_len, d_model)

        # Positions: [0, 1, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # Define frequency terms (logarithmically spaced)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Assign sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: shape becomes [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer so it's not a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding up to the length of x (seq_len)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Transformer model for time series prediction.

    Includes positional encoding, causal masking, and a single encoder block.
    """
    def __init__(self, input_size, d_model, output_size, dropout_rate=0.3,
                 nhead=2, num_layers=1, dim_feedforward_factor=4, max_seq_length=500):
        super(TransformerModel, self).__init__()

        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)

        # Add positional encoding to retain sequence information
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate, max_seq_length)

        # Define a single Transformer encoder layer with LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * dim_feedforward_factor,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True  
        )

        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # Output prediction layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)

    def forward(self, x):
        # Project input to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through Transformer encoder
        x = self.transformer_encoder(x)

        # Final normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Select the last time step's output as sequence summary
        x = x[:, -1, :]

        # Pass through fully connected prediction layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class CNN_LSTM(nn.Module):
    """
    CNN is used to extract features from input time series
    LSTM is used to process these features for prediction
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, 
                 output_size, dropout_rate, num_filters_l1, num_filters_l2, 
                 num_filters_l3, kernel_size, pool_kernel_size=2, 
                 pool_stride=1, pool_padding=1):
        super(CNN_LSTM, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters_l1, 
                              kernel_size=kernel_size, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=pool_kernel_size, 
                                  stride=pool_stride, 
                                  padding=pool_padding)
        self.dropout_cnn1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(in_channels=num_filters_l1, out_channels=num_filters_l2, 
                              kernel_size=kernel_size, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=pool_kernel_size, 
                                 stride=pool_stride, 
                                 padding=pool_padding)
        self.dropout_cnn2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv1d(in_channels=num_filters_l2, out_channels=num_filters_l3, 
                              kernel_size=kernel_size, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=pool_kernel_size, 
                                 stride=pool_stride, 
                                 padding=pool_padding)
        self.dropout_cnn3 = nn.Dropout(dropout_rate)
        
        # First LSTM layer and its auxiliary layers
        self.lstm1 = nn.LSTM(num_filters_l3, hidden_size1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second LSTM layer and its auxiliary layers
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third LSTM layer and its auxiliary layers
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size3, hidden_size3 // 2)
        self.fc2 = nn.Linear(hidden_size3 // 2, output_size)
    
    def _process_lstm_layer(self, x, lstm_layer, bn_layer, dropout_layer):
        """Helper method to process data through one LSTM layer"""
        x, _ = lstm_layer(x)
        x = x.permute(0, 2, 1)
        x = bn_layer(x)
        x = x.permute(0, 2, 1)
        x = dropout_layer(x)
        return x
        
    def forward(self, x):
        # Process through CNN layers
        # Reshape for CNN (batch_size, input_size, seq_len)
        x_cnn = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x_cnn = F.relu(self.conv1(x_cnn))
        x_cnn = self.pool1(x_cnn)
        x_cnn = self.dropout_cnn1(x_cnn)
        
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = self.pool2(x_cnn)
        x_cnn = self.dropout_cnn2(x_cnn)
        
        x_cnn = F.relu(self.conv3(x_cnn))
        x_cnn = self.pool3(x_cnn)
        x_cnn = self.dropout_cnn3(x_cnn)
        
        # Reshape back for LSTM (batch_size, seq_len, input_size)
        x_cnn = x_cnn.permute(0, 2, 1)
        
        # Process through LSTM layers
        x = self._process_lstm_layer(x_cnn, self.lstm1, self.bn1, self.dropout1)
        x = self._process_lstm_layer(x, self.lstm2, self.bn2, self.dropout2)
        x = self._process_lstm_layer(x, self.lstm3, self.bn3, self.dropout3)
        
        # Select the output of the last time step
        x = x[:, -1, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class AttentionMechanism(nn.Module):
    """Attention mechanism implementation for Attention-LSTM model.
    
    This model implements the attention mechanism that can focus on relevant
    parts of the input sequence.
    """
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        
        # Attention layers
        self.attn_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_score = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: Current decoder hidden state, shape [batch_size, hidden_size]
            encoder_outputs: All encoder outputs, shape [batch_size, seq_len, hidden_size]
        
        Returns:
            context_vector: Weighted sum of encoder outputs based on attention scores
            attention_weights: Attention weights for visualization
        """
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        
        # Expand hidden state to match encoder_outputs time steps
        hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate encoder outputs and hidden state
        # [batch_size, seq_len, hidden_size * 2]
        attn_input = torch.cat((encoder_outputs, hidden_expanded), dim=2)
        
        # Calculate attention scores
        # [batch_size, seq_len, hidden_size]
        attn_hidden = torch.tanh(self.attn_hidden(attn_input))
        
        # [batch_size, seq_len, 1]
        score = self.attn_score(attn_hidden)
        
        # [batch_size, seq_len]
        attention_weights = F.softmax(score.squeeze(2), dim=1)
        
        # Create context vector by applying attention weights to encoder outputs
        # [batch_size, 1, seq_len] * [batch_size, seq_len, hidden_size]
        # -> [batch_size, 1, hidden_size]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        # [batch_size, hidden_size]
        context_vector = context_vector.squeeze(1)
        
        return context_vector, attention_weights

class AttentionLSTM(nn.Module):
    """Attention-LSTM model implementation for water temperature prediction.
    
    This model combines LSTM layers with attention mechanism to better focus on
    relevant parts of historical weather data for predicting water temperature.
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, 
                 output_size, dropout_rate):
        super(AttentionLSTM, self).__init__()
        
        # First LSTM layer and its auxiliary layers
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second LSTM layer and its auxiliary layers
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third LSTM layer and its auxiliary layers
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Attention mechanism
        self.attention = AttentionMechanism(hidden_size3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size3, hidden_size3 // 2)
        self.fc2 = nn.Linear(hidden_size3 // 2, output_size)
    
    def _process_lstm_layer(self, x, lstm_layer, bn_layer, dropout_layer):
        """Helper method to process data through one LSTM layer"""
        x, _ = lstm_layer(x)
        x = x.permute(0, 2, 1)
        x = bn_layer(x)
        x = x.permute(0, 2, 1)
        x = dropout_layer(x)
        return x
        
    def forward(self, x):
        """
        Args:
            x: Historical time series data [batch_size, seq_len, input_size]
                         
        Returns:
            output: Predicted values [batch_size, output_size]
        """
        # Process historical input through LSTM layers
        x = self._process_lstm_layer(x, self.lstm1, self.bn1, self.dropout1)
        x = self._process_lstm_layer(x, self.lstm2, self.bn2, self.dropout2)
        encoded_sequence = self._process_lstm_layer(x, self.lstm3, self.bn3, self.dropout3)
        
        # Get the last hidden state as the decoder initial state
        decoder_hidden = encoded_sequence[:, -1, :]
        
        # Apply attention to find relevant parts in the sequence
        context_vector, _ = self.attention(decoder_hidden, encoded_sequence)
        
        # Fully connected layers
        x = F.relu(self.fc1(context_vector))
        output = self.fc2(x)
        
        return output

