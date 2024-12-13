# Comparing Graph Neural Networks Against Transformer Based Graph Learning

## About
This project aims to build and test different types of Sequence-to-Sequence (Seq2Seq) architectures and compare their performance on a translation task (English to Japanese). Using this exploration, I aim to identify the strengths and weaknesses of different Seq2Seq architectures, which can provide better understanding of similar natural language processing tasks. This exploration is motivated by the increasing importance of communication in an ever growing and connected world. The 4 encoder/decoder structures tested are Gated Recurrent Unit (GRU), Long Short Term Memory (LSTM), Transformers, and Convolutional Neural Networks (CNN). And the data used is the Japanese-English Subtitle Corpus (JESC). In this project, I show that CNNs performed the strongest out of the 4 models due to its ability to converge faster and its ability to capture local connections when translating individual sentences.

## Set Up
### Set Up Environment
The conda environment needed to run the code can be found in `environment.yml`. You can initialize the conda environment using the following command: `conda env create -f environment.yml`

### Download Data
You can download the cleaned data set for the [Stanford JESC Project Data](https://nlp.stanford.edu/projects/jesc/) from [this Kaggle link](https://www.kaggle.com/datasets/wahyusetianto/japanesse-english-subtittle-corpusjesc-cleaned). Make sure to save the folder as `data_jesc/`.

## Code
Saved models and vocabularies (Eng and Jpn) will be in a directory called `saved/`.

### etl.py
Contains the code for downloading the data, data preprocessing, model definitions, and model training.
#### Vocabulary Management
- **`build_vocab(tokenized_data, special_tokens, max_vocab_size=5000) -> dict`**  
  Builds a vocabulary dictionary from tokenized data, adds special tokens, and limits the vocabulary size.
- **`safe_tokens_to_indices(tokens, vocab, sos_eos=True) -> list`**  
  Converts tokens into their corresponding indices using a vocabulary. Adds `<START>` and `<END>` tokens if `sos_eos=True`.
- **`pad_sequence(sequence, max_len, pad_value=0) -> list`**  
  Pads or truncates a sequence of token indices to a fixed length (`max_len`) using a specified padding value (`pad_value`).

#### File Operations
- **`save_vocab(vocab, path) -> None`**  
  Saves a vocabulary dictionary to a JSON file at the specified path.
- **`load_vocab(path) -> dict`**  
  Loads a vocabulary dictionary from a JSON file at the specified path.

#### Data Loading and Preprocessing
- **`load_and_process_data(max_vocab=5000, subset_train=1.0, subset_test=1.0, subset_val=1.0)`**  
  Preprocesses English and Japanese sentence pairs from datasets, generates tokenized sequences, and saves vocabularies to disk.  
  **Returns**: Train, test, and validation DataLoaders, and English/Japanese vocabularies.

#### Training and Evaluation
- **`train_model(model, train_loader, optimizer, criterion, device, padding_idx=0) -> (float, float)`**  
  Trains the model on the training data and calculates average loss and accuracy.  
  **Returns**:  
  - `float`: Average training loss.  
  - `float`: Average training accuracy.

- **`evaluate_model(model, test_loader, criterion, device, padding_idx=0, end_token_idx=2) -> (float, float)`**  
  Evaluates the model on test data and calculates average loss and accuracy.  
  **Returns**:  
  - `float`: Average test loss.  
  - `float`: Average test accuracy.
#### Dataset Class
- **`TranslationDataset(src_data, tgt_data)`**  
  Custom Dataset class for source and target sequences.  
  **Methods**:  
  - `__len__() -> int`: Returns the size of the dataset.  
  - `__getitem__(idx) -> (torch.Tensor, torch.Tensor)`: Returns a source-target pair as tensors.

#### Models
#### Sequence-to-Sequence Architectures
- **`LSTMSeq2Seq(input_dim, output_dim, embed_dim, hidden_dim, n_layers, dropout)`**  
  LSTM-based encoder-decoder model.
- **`GRUSeq2Seq(input_dim, output_dim, embed_dim, hidden_dim, n_layers, dropout)`**  
  GRU-based encoder-decoder model.
- **`CNNSeq2Seq(input_dim, output_dim, embed_dim, kernel_size, num_channels)`**  
  CNN-based encoder-decoder model.
- **`TransformerSeq2Seq(input_dim, output_dim, embed_dim, n_heads, num_layers, dropout)`**  
  Transformer-based encoder-decoder model.
####  Saving and Loading Models
- **`save_model_and_params(models, save_dir)`**  
  Saves model parameters and weights to a specified directory.  
  **Inputs**:  
  - `models`: Dictionary of model names and instances.  
  - `save_dir`: Directory path to save the models.
- **`load_model(model_name, save_dir, device) -> nn.Module`**  
  Dynamically loads a model based on saved parameters and weights.  
  **Inputs**:  
  - `model_name`: Name of the model to load.  
  - `save_dir`: Directory containing the saved model files.  
  - `device`: Device to load the model onto.  
  **Returns**: The loaded model instance.
#### Running the Main Script
- **`main()`**  
  Runs the full pipeline of data preprocessing, training, and evaluation for all models.  
  **Outputs**:  
  - Processed datasets.  
  - Trained models saved to the `saved/` directory.  
  - Test results for all models
### translate.py
Contains the translation function using a saved model and an input English sentence.
- **translate_sentence(sentence, model_name = 'CNN_Seq2Seq', save_dir = "saved")**
### testing.ipynb
Shows the code for training and testing. Here you can see the models I trained and tested (including train-test accuracies) and how to also use the code in a .ipynb setting.



