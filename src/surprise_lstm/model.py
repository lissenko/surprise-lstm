import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .features import get_input_size, extract_melody_notes, get_note_vec

class SurpriseLSTM(nn.Module):
    def __init__(self, features, num_pitches=128, hidden_size=256, num_layers=3, dropout=0.5):
        super(SurpriseLSTM, self).__init__()
        self.features = features
        self.num_pitches = num_pitches
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=get_input_size(features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.pitch_out = nn.Linear(hidden_size, num_pitches)
        
        self.clip_norm = 5
        
    def forward(self, x, masks, hidden=None):
        lengths = masks.sum(dim=1).cpu().long()
        
        packed_input = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_output, hidden = self.lstm(packed_input, hidden)
        
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        output = self.dropout(output)
        
        pitch_logits = self.pitch_out(output)
        
        return pitch_logits, hidden

def predict(model, midi_path, device='cuda'):
    if not os.path.isfile(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    max_duration, max_onset, max_ioi = model.max_duration, model.max_onset, model.max_ioi
    reference_melody = extract_melody_notes(midi_path, min_notes=0)[0]

    model.eval()
    ics = []
    pitch_distributions = []
    entropies = []
    durations = []
    losses = []

    input_dim = get_input_size(model.features)
    current_seq = torch.zeros(1, input_dim).to(device)

    hidden = None

    with torch.no_grad():
        for idx, note_representation in enumerate(reference_melody):
            mask = torch.ones(1, len(current_seq), dtype=torch.float32).to(device)
            
            logits, hidden = model(current_seq.unsqueeze(0), mask, None)
            logits = logits[0, -1]  # Last timestep logits

            probs = torch.softmax(logits, dim=-1)
            pitch_distributions.append(probs)
            true_pitch = note_representation['pitch']
            true_duration = note_representation['duration']

            ic = -np.log2(probs[true_pitch].item())
            ics.append(ic)
            H = -(probs * torch.log2(probs)).sum().item()
            entropies.append(H)
            durations.append(true_duration)

            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([true_pitch], device=device))
            losses.append(loss.item())

            new_note = get_note_vec(note_representation, max_duration, max_onset, max_ioi, model.features).to(device)
            current_seq = torch.cat([current_seq, new_note.unsqueeze(0)])

        # Extra note
        mask = torch.ones(1, len(current_seq), dtype=torch.float32, device=device)
        logits, _ = model(current_seq.unsqueeze(0), mask, None)
        logits = logits[0, -1]
        last_note_probs = torch.softmax(logits, dim=-1)

    avg_test_loss = sum(losses) / len(losses) if losses else float('nan')
    mdwics = sum(ic_i * dur_i for ic_i, dur_i in zip(ics, durations)) / sum(durations)

    return {'ics': ics, 'entropies': entropies, 'pitch_distributions':
            pitch_distributions, 'mdwics': mdwics, 'avg_test_loss':
            avg_test_loss, 'extra_note_distrib': last_note_probs}

def load_model_from_checkpoint(path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SurpriseLSTM(
        ckpt["features"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
        num_pitches=ckpt.get("num_pitches", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.max_duration = ckpt["max_duration"]
    model.max_onset = ckpt["max_onset"]
    model.max_ioi = ckpt["max_ioi"]
    return model, device, ckpt
