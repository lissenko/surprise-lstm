# surprise-lstm

LSTM-based model of melodic surprise from MIDI.

## Installation

```bash
git clone https://github.com/lissenko/surprise-lstm.git
cd surprise-lstm
pip install -r requirements.txt
pip install -e .
```

## Pretrained model

The default model was trained on [Tegridy-MIDI-Dataset/Clean-Melodies](https://github.com/asigalov61/Tegridy-MIDI-Dataset/tree/master/Clean-Melodies).

[Download](https://github.com/lissenko/surprise-lstm/releases/download/v0.1.0/default_model.pth)

## Quick Start

```py
from surprise_lstm import load_model_from_checkpoint, predict

model, device, _ = load_model_from_checkpoint('default_model.pth')
midi_file = "/path/to/your/file"
result = predict(model, midi_file)

ics = result['ics']
entropies = result['entropies']
pitch_distributions = result['pitch_distributions']
mdwics = result['mdwics']
avg_test_loss = result['avg_test_loss']
extra_note_distribution = result['extra_note_distrib']
```

- `ics`: list of per-note information content (bits).
- `entropies`: list of per-note entropy from the predicted pitch distribution at each step.
- `pitch_distributions`: list of length-128 arrays; each is the model’s pitch probability distribution for the corresponding note.
- `mdwics`: mean duration-weighted IC over the melody.
- `avg_test_loss`: mean cross-entropy loss.
- `extra_note_distrib`: length-128 probability vector for the imaginary next note after the melody.

## Citing

```
@mastersthesis{lissenko2025,
  title        = {Computational Modeling of Musical Surprise: Deep Learning Estimation from Melodies},
  author       = {Tanguy Lissenko},
  school       = {Music Technology Group, Master's in Sound and Music Computing},
  year         = {2025},
  month        = aug,
  type         = {Master's Thesis},
  Supervisors  = {Martin Rocamora, Manuel Anglada-Tort},
}
```
