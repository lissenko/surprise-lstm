from .model import SurpriseLSTM, predict, load_model_from_checkpoint
from .features import get_input_size, extract_melody_notes, get_note_vec

__all__ = [
    "SurpriseLSTM",
    "predict",
    "load_model_from_checkpoint",
    "get_input_size",
    "extract_melody_notes",
    "get_note_vec",
]

__version__ = "0.2.0"
