import torch
from nltk.tokenize import word_tokenize  # Ensure NLTK is installed
import torch
import argparse
from etl import * 

### GLOBAL VARIABLES ###
SEED = 28
SPECIAL_TOKENS = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNKNOWN>': 3}
MAX_LEN = 17
### END OF GLOBAL VARIABLES ###

# set random seed
random.seed(SEED)
torch.manual_seed(SEED)  # Seed for CPU computations
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)  # Seed for GPU computations


def translate_sentence(sentence, model_name = 'CNN_Seq2Seq', save_dir = "saved"):
    """
    Translates a sentence using the given model.
    Args:
        sentence (str): The input sentence to translate.
        model_name: The name of the trained translation model
    Returns:
        str: Translated sentence.
    """
    # load in the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_name, save_dir, device)
    # Ensure the model is in evaluation mode
    model.eval()

    # load in vocab
    src_vocab = load_vocab('saved/eng_vocab.json')
    tgt_vocab = load_vocab('saved/jpn_vocab.json')
    max_len = MAX_LEN

    # Tokenize the input sentence
    tokens = word_tokenize(sentence)

    src_indices = [src_vocab.get(token, src_vocab['<UNKNOWN>']) for token in tokens]
    src_indices = [src_vocab['<START>']] + src_indices + [src_vocab['<END>']]
    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)
    tgt_indices = [tgt_vocab['<START>']]
    tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long, device=device)

    for _ in range(max_len):
        # Generate predictions
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        next_token = output[:, -1, :].argmax(dim=-1).item()
        # Stop if <END> token is predicted
        if next_token == tgt_vocab['<END>']:
            break
        # Add the predicted token to the target tensor
        tgt_indices.append(next_token)
        tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long, device=device)

    # Convert target indices to tokens
    tgt_vocab_inv = {idx: token for token, idx in tgt_vocab.items()}
    translated_tokens = [tgt_vocab_inv[idx] for idx in tgt_indices[1:]]  # Skip the <START> token
    return ' '.join(translated_tokens)

def main(sentence):
    print(translate_sentence(sentence))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with inputs.")
    parser.add_argument('--sentence', type=str, required=True, help="English sentence to translate.")
    # Parse the command-line arguments
    args = parser.parse_args()
    # Pass the arguments to the main function
    main(args.sentence)