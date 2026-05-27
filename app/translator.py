import torch
import sentencepiece as spm
from app.model import Encoder, Decoder, Seq2Seq, PAD_IDX, BOS_IDX, EOS_IDX

# model hyperparameters - must match the saved checkpoint
VOCAB_SIZE = 16000
EMB_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.3


def load_model(checkpoint_path: str, spm_model_path: str, device: torch.device):
    """Load the trained model and tokenizer from disk."""
    encoder = Encoder(VOCAB_SIZE, EMB_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    decoder = Decoder(VOCAB_SIZE, EMB_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)

    return model, sp


def translate(sentence: str, model: Seq2Seq, sp, device: torch.device, max_len: int = 50) -> str:
    """Translate a single English sentence to Uzbek."""
    model.eval()

    # tokenize input
    tokens = [BOS_IDX] + sp.encode(sentence.lower().strip()) + [EOS_IDX]
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    src_lengths = torch.tensor([len(tokens)], dtype=torch.long)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_lengths)

    # decode one token at a time until EOS or max_len
    trg_ids = [BOS_IDX]
    for _ in range(max_len):
        input_token = torch.tensor([trg_ids[-1]]).to(device)
        with torch.no_grad():
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
        pred_id = output.argmax(1).item()
        trg_ids.append(pred_id)
        if pred_id == EOS_IDX:
            break

    # strip BOS/EOS and decode
    trg_ids = trg_ids[1:]
    if EOS_IDX in trg_ids:
        trg_ids = trg_ids[:trg_ids.index(EOS_IDX)]

    return sp.decode(trg_ids)
