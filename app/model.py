import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Token IDs (must match what was used during training)
PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3


class Encoder(nn.Module):
    """Bidirectional GRU encoder. Reads the English sentence and produces context vectors."""

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            emb_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        # merge forward + backward hidden into one vector
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.emb(src))
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        # combine last forward and backward hidden states
        hidden = hidden.view(self.num_layers, 2, src.size(0), self.hidden_size)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)))

        return outputs, hidden


class Decoder(nn.Module):
    """GRU decoder with Bahdanau-style attention. Generates Uzbek tokens one by one."""

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=emb_size + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size + hidden_size * 2 + emb_size, vocab_size)

        # attention layers
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size * 2, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, input_token, hidden, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.emb(input_token))

        # attention: score each encoder output against current hidden state
        scores = self.Va(torch.tanh(self.Wa(hidden.unsqueeze(1)) + self.Ua(encoder_outputs)))
        attention_weights = torch.softmax(scores, dim=1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1, keepdim=True)

        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))

        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        hidden = hidden.squeeze(0)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden, attention_weights


class Seq2Seq(nn.Module):
    """Full encoder-decoder model wrapper."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
