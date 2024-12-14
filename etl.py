import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from collections import Counter
from itertools import chain
from nltk.tokenize import word_tokenize  # Ensure NLTK is installed
from sklearn.metrics import accuracy_score
import os
import json
import torch
import random

### GLOBAL VARIABLES ###
SEED = 28
SPECIAL_TOKENS = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNKNOWN>': 3}
MAX_LEN = 17
### END OF GLOBAL VARIABLES ###

# set random seeds
random.seed(SEED)
torch.manual_seed(SEED)  
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  

### DATA PREPROCESSING ###
def build_vocab(tokenized_data, special_tokens, max_vocab_size=5000):
    """ 
    builds eng of jpn vocab using tokenized data and adds new special tokens 
    """
    tokenized_data = tokenized_data.explode() 
    vocab_counter = Counter(tokenized_data)
    most_common = vocab_counter.most_common(max_vocab_size) 
    vocab = {word: idx + len(special_tokens) for idx, (word, _) in enumerate(most_common)}
    vocab.update(special_tokens)
    return vocab

def safe_tokens_to_indices(tokens, vocab, sos_eos=True):
    """ 
    turn tokens into numerical indicies (unique)
    """
    indices = [vocab.get(token, vocab['<UNKNOWN>']) for token in tokens]
    if sos_eos:
        indices = [vocab['<START>']] + indices + [vocab['<END>']]
    return indices

def pad_sequence(sequence, max_len, pad_value=0):
    """
    pad sequences to certain length
    """
    return sequence[:max_len] + [pad_value] * max(0, max_len - len(sequence))

def save_vocab(vocab, path):
    """
    Save a vocabulary dictionary to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)  

def load_vocab(path):
    """
    Load a vocabulary dictionary from a JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)
    
def load_and_process_data(max_vocab = 5000, subset_train = 1.0, subset_test = 1.0, subset_val = 1.0):
    """ 
    Load in data
    Gets data and saves preprocessed eng_vocab and jpn_vocab into files in directory /saved/
    """
    test_data = pd.read_csv('data_jesc/test', sep='\t')
    train_data = pd.read_csv('data_jesc/train', sep='\t')
    val_data = pd.read_csv('data_jesc/dev', sep = '\t')
    print('Loading Data')
    train_data = train_data.sample(frac=1, random_state=42).head(int(train_data.shape[0] * subset_train))
    test_data = test_data.sample(frac=1, random_state=42).head(int(test_data.shape[0] * subset_test))
    val_data = val_data.sample(frac=1, random_state=42).head(int(val_data.shape[0] * subset_val))
    test_data.columns = ['ENG', 'JPN']
    train_data.columns = ['ENG', 'JPN']
    val_data.columns = ['ENG', 'JPN']
    # tokenize
    test_data['ENG'] = test_data['ENG'].apply(word_tokenize)
    test_data['JPN'] = test_data['JPN'].apply(word_tokenize)
    train_data['ENG'] = train_data['ENG'].apply(word_tokenize)
    train_data['JPN'] = train_data['JPN'].apply(word_tokenize)
    val_data['ENG'] = val_data['ENG'].apply(word_tokenize)
    val_data['JPN'] = val_data['JPN'].apply(word_tokenize)
    # get sentence lengths
    all_lengths = pd.concat([
        train_data['ENG'].apply(len),
        train_data['JPN'].apply(len),
        test_data['ENG'].apply(len),
        test_data['JPN'].apply(len),
        val_data['ENG'].apply(len),
        val_data['JPN'].apply(len)
    ])
    # 95th percentile for MAX_LEN 
    MAX_LEN = int(all_lengths.quantile(0.95))
    eng_vocab = build_vocab(train_data['ENG'], SPECIAL_TOKENS, max_vocab_size=max_vocab)
    jpn_vocab = build_vocab(train_data['JPN'], SPECIAL_TOKENS, max_vocab_size=max_vocab)
    # preprocess data
    train_data['ENG'] = train_data['ENG'].apply(lambda x: pad_sequence(safe_tokens_to_indices(x, eng_vocab), MAX_LEN))
    train_data['JPN'] = train_data['JPN'].apply(lambda x: pad_sequence(safe_tokens_to_indices(x, jpn_vocab), MAX_LEN))
    test_data['ENG'] = test_data['ENG'].apply(lambda x: pad_sequence(safe_tokens_to_indices(x, eng_vocab), MAX_LEN))
    test_data['JPN'] = test_data['JPN'].apply(lambda x: pad_sequence(safe_tokens_to_indices(x, jpn_vocab), MAX_LEN))
    val_data['ENG'] = val_data['ENG'].apply(lambda x: pad_sequence(safe_tokens_to_indices(x, eng_vocab), MAX_LEN))
    val_data['JPN'] = val_data['JPN'].apply(lambda x: pad_sequence(safe_tokens_to_indices(x, jpn_vocab), MAX_LEN))
    # save vocabularies
    save_vocab(eng_vocab, 'saved/eng_vocab.json')
    save_vocab(jpn_vocab, 'saved/jpn_vocab.json')
    # turn into dataset obj
    train_dataset = TranslationDataset(train_data['ENG'].tolist(), train_data['JPN'].tolist())
    test_dataset = TranslationDataset(test_data['ENG'].tolist(), test_data['JPN'].tolist())
    val_dataset = TranslationDataset(val_data['ENG'].tolist(), val_data['JPN'].tolist())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print("Vocabularies saved successfully!")
    return train_loader, test_loader, val_loader, eng_vocab, jpn_vocab
### DATA PREPROCESSING END ###

# train and test
def train_model(model, train_loader, optimizer, criterion, device, padding_idx=0):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])  # Predict target sequence excluding the last token
        if output.size(1) != tgt[:, 1:].size(1):
            min_seq_len = min(output.size(1), tgt[:, 1:].size(1))
            output = output[:, :min_seq_len, :]
            tgt = tgt[:, :min_seq_len + 1]
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        preds = output.argmax(dim=-1)  
        mask = tgt[:, 1:] != padding_idx  
        correct = ((preds == tgt[:, 1:]) & mask).float()  
        total_correct += correct.sum().item()
        total_samples += mask.sum().item()  
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, avg_accuracy

def evaluate_model(model, test_loader, criterion, device, padding_idx=0, end_token_idx = 2):
    """
    Evaluates the model on a test dataset while skipping padding tokens in accuracy calculation.
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1]) 
            if output.size(1) != tgt[:, 1:].size(1):
                min_seq_len = min(output.size(1), tgt[:, 1:].size(1))
                output = output[:, :min_seq_len, :]
                tgt = tgt[:, :min_seq_len + 1]
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            preds = output.argmax(dim=-1) 
            mask = (tgt[:, 1:] != padding_idx) & (tgt[:, 1:] != end_token_idx) 
            correct = ((preds == tgt[:, 1:]) & mask).float() 
            total_correct += correct.sum().item()
            total_samples += mask.sum().item() 
    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, avg_accuracy

### CLASS DEFINITIONS ###
# Dataset Class
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
    def __len__(self):
        return len(self.src_data)
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long), torch.tensor(self.tgt_data[idx], dtype=torch.long)

### MODELS ###
# LSTM Seq2Seq Model
class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, n_layers, dropout):
        super(LSTMSeq2Seq, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.embedding_src = nn.Embedding(input_dim, embed_dim)
        self.embedding_tgt = nn.Embedding(output_dim, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    def forward(self, src, tgt):
        src_embedded = self.embedding_src(src)
        _, (hidden, cell) = self.encoder(src_embedded)
        tgt_embedded = self.embedding_tgt(tgt)
        outputs, _ = self.decoder(tgt_embedded, (hidden, cell))
        return self.fc_out(outputs)

# CNN Seq2Seq Model
class CNNSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, kernel_size, num_channels):
        super(CNNSeq2Seq, self).__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_src = nn.Embedding(input_dim, embed_dim)
        self.embedding_tgt = nn.Embedding(output_dim, embed_dim)
        self.encoder = nn.Conv1d(embed_dim, num_channels, kernel_size, padding=kernel_size // 2)
        self.decoder = nn.Conv1d(num_channels + embed_dim, num_channels, kernel_size, padding=kernel_size // 2)
        self.fc_out = nn.Linear(num_channels, output_dim)
    def forward(self, src, tgt):
        src_embedded = self.embedding_src(src).permute(0, 2, 1)  
        tgt_embedded = self.embedding_tgt(tgt).permute(0, 2, 1)  
        encoder_outputs = self.encoder(src_embedded)
        tgt_embedded = tgt_embedded[:, :, :encoder_outputs.size(2)] 
        if tgt_embedded.size(2) < encoder_outputs.size(2):
            pad_size = encoder_outputs.size(2) - tgt_embedded.size(2)
            tgt_embedded = torch.nn.functional.pad(tgt_embedded, (0, pad_size))
        decoder_inputs = torch.cat((encoder_outputs, tgt_embedded), dim=1) 
        decoder_outputs = self.decoder(decoder_inputs) 
        return self.fc_out(decoder_outputs.permute(0, 2, 1))

# Transformer Seq2Seq Model
class TransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, n_heads, num_layers, dropout):
        super(TransformerSeq2Seq, self).__init__()
        self.embedding_src = nn.Embedding(input_dim, embed_dim)
        self.embedding_tgt = nn.Embedding(output_dim, embed_dim)
        self.transformer = nn.Transformer(embed_dim, n_heads, num_layers, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed_dim = embed_dim
    def forward(self, src, tgt):
        src_embedded = self.embedding_src(src).permute(1, 0, 2)  
        tgt_embedded = self.embedding_tgt(tgt).permute(1, 0, 2)  
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embedded.size(0)).to(src.device)
        output = self.transformer(src_embedded, tgt_embedded, tgt_mask=tgt_mask)
        return self.fc_out(output.permute(1, 0, 2))  

class GRUSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, n_layers, dropout):
        """
        GRU-based Seq2Seq model
        """
        super(GRUSeq2Seq, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.embed_dim = embed_dim 
        self.hidden_dim = hidden_dim 
        self.n_layers = n_layers 
        self.dropout = dropout
        self.embedding_src = nn.Embedding(input_dim, embed_dim)
        self.embedding_tgt = nn.Embedding(output_dim, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, tgt):
        """
        Forward pass for GRU-based Seq2Seq model.
        """
        src_embedded = self.dropout(self.embedding_src(src)) 
        tgt_embedded = self.dropout(self.embedding_tgt(tgt))
        _, hidden = self.encoder(src_embedded) 
        decoder_outputs, _ = self.decoder(tgt_embedded, hidden)
        outputs = self.fc_out(decoder_outputs) 
        return outputs
    
### END OF CLASS DEFINITIONS ###

### SAVING AND LOADING MODELS ###
def save_model_and_params(models, save_dir):
    """
    Save models and their parameters.
    """
    os.makedirs(save_dir, exist_ok=True)
    for name, model in models.items():
        if isinstance(model, TransformerSeq2Seq):
            params = {
                "model_type": "Transformer",
                "input_dim": model.input_dim,
                "output_dim": model.output_dim,
                "embed_dim": model.fc_out.in_features,
                "n_heads": model.n_heads,
                "num_layers": model.num_layers,
                "dropout": model.dropout
            }
        elif isinstance(model, CNNSeq2Seq):
            params = {
                "model_type": "CNN",
                "input_dim": model.input_dim,
                "output_dim": model.output_dim,
                "embed_dim": model.embed_dim,
                "kernel_size": model.kernel_size,
                "num_channels": model.num_channels
            }
        elif isinstance(model, GRUSeq2Seq):
            params = {
                "model_type": "GRU",
                "input_dim": model.input_dim,
                "output_dim": model.output_dim,
                "embed_dim": model.embed_dim,
                "hidden_dim": model.hidden_dim,
                "n_layers": model.n_layers,
                "dropout": model.dropout.p if isinstance(model.dropout, torch.nn.Dropout) else model.dropout
            }
        elif isinstance(model, LSTMSeq2Seq):
            params = {
                "model_type": "LSTM",
                "input_dim": model.input_dim,
                "output_dim": model.output_dim,
                "embed_dim": model.embed_dim,
                "hidden_dim": model.hidden_dim,
                "n_layers": model.n_layers,
                "dropout": model.dropout
            }
        else:
            raise ValueError(f"Unknown model type for {name}")
        params_path = os.path.join(save_dir, f"{name}_params.json")
        with open(params_path, 'w') as f:
            json.dump(params, f)
        # save model weights
        weights_path = os.path.join(save_dir, f"{name}_weights.pt")
        torch.save(model.state_dict(), weights_path)
        print(f"Saved {name} to {params_path} and {weights_path}.")

def load_model_params(params_path):
    """
    loads model's params, helper func
    """
    with open(params_path, 'r') as f:
        return json.load(f)

def load_model(model_name, save_dir, device):
    """
    loads a model
    """
    params_path = os.path.join(save_dir, f"{model_name}_params.json")
    weights_path = os.path.join(save_dir, f"{model_name}_weights.pt")
    params = load_model_params(params_path)
    if params["model_type"] == "Transformer":
        model = TransformerSeq2Seq(
            params["input_dim"],
            params["output_dim"],
            params["embed_dim"],
            params["n_heads"],
            params["num_layers"],
            params["dropout"]
        )
    elif params["model_type"] == "GRU":
        model = GRUSeq2Seq(
            params["input_dim"],
            params["output_dim"],
            params["embed_dim"],
            params["hidden_dim"],
            params["n_layers"],
            params["dropout"]
        )
    elif params["model_type"] == "CNN":
        model = CNNSeq2Seq(
            params["input_dim"],
            params["output_dim"],
            params["embed_dim"],
            params["kernel_size"],
            params["num_channels"]
        )
    elif params["model_type"] == "LSTM":
        model = LSTMSeq2Seq(
            params["input_dim"],
            params["output_dim"],
            params["embed_dim"],
            params["hidden_dim"],
            params["n_layers"],
            params["dropout"]
        )
    else:
        raise ValueError(f"Unknown model type: {params['model_type']}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    print(f"Loaded {model_name} ({params['model_type']}) from {weights_path}.")
    return model
### END OF SAVING AND LOADING MODELS ###

### TRAIN ALL MODELS ###
def train_all_models(eng_vocab, jpn_vocab, train_data, test_data):
    """ 
    Trains all models like in testing.ipynb
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = TranslationDataset(train_data['ENG'].tolist(), train_data['JPN'].tolist())
    test_dataset = TranslationDataset(test_data['ENG'].tolist(), test_data['JPN'].tolist())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # models dictionary
    models = {
    'Transformer_Seq2Seq': TransformerSeq2Seq(len(eng_vocab), len(jpn_vocab), 15, 3, 3, 0.2),
    'LSTM_Seq2Seq': LSTMSeq2Seq(len(eng_vocab), len(jpn_vocab), 15, 10, 3, 0.2),
    'GRU_Seq2Seq': GRUSeq2Seq(len(eng_vocab), len(jpn_vocab), 15, 5, 3, 0.2),
    'CNN_Seq2Seq': CNNSeq2Seq(len(eng_vocab), len(jpn_vocab), 15, kernel_size=3, num_channels=5)
    }
    # save and report results
    results = {}
    EPOCHS = 50
    # train all the models
    for name, model in models.items():
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-5)
        criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS['<PAD>'])
        print(f"Training {name}...")
        for epoch in range(EPOCHS):  
            train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device)
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}")
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"{name} Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")
        results[name] = (test_loss, test_accuracy)
    print("\nModel Comparison:")
    for model_name, (loss, accuracy) in results.items():
        print(f"{model_name}: Test Loss = {loss:.4f}, Test Accuracy = {accuracy:.4f}")
    print('\nSaving Models')
    # save models and parameters
    save_dir = "saved"
    save_model_and_params(models, save_dir)
    return results

def main():
    train_data, test_data, eng_vocab, jpn_vocab = load_and_process_data()
    results = train_all_models(eng_vocab, jpn_vocab, train_data, test_data)


if __name__ == "__main__":
    main()