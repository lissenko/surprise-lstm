from music21 import stream, note, analysis
import numpy as np
import pretty_midi
import torch

SIXTY_FOURTH_DURATION = 0.0625
PHRASE_IOI_THRESHOLD = 1.5  # (beats)
UNDEFINED = None

FEATURE_DIM = {
        'pitch': 128,
        'duration': 1,
        'symbolic_duration': 21,
        'interval': 73, # Covers intervals from -36 to +36 
        'onset': 1,
        'contour': 3,
        'pitch_class': 12,
        'beat_position': 1,
        'ioi': 1,
        'scale_degree': 7,
        'key_membership': 1,
        'is_repeated_pitch': 1,
        'register': 3,
        'phrase': 2,
        'cpintfip': 73,
        }

def get_input_size(features_list):
    size = 0
    for feature in features_list:
        size += FEATURE_DIM[feature]
    return size

def get_note_type(duration_ratio):
    base_durations = [
        4.0,       # whole
        2.0,       # half
        1.0,       # quarter
        0.5,       # eighth
        0.25,      # sixteenth
        0.125,     # thirty-second
        0.0625     # sixty-fourth
    ]

    all_durations = []
    for base in base_durations:
        all_durations.append(base)             # plain
        all_durations.append(base * 1.5)       # dotted
        all_durations.append(base * 1.75)      # double dotted

    # Find index of the closest match
    closest_index = min(
        range(len(all_durations)),
        key=lambda i: abs(all_durations[i] - duration_ratio)
    )
    return closest_index

def is_strictly_monophonic(notes):
    if len(notes) < 1:
        return False
    notes = sorted(notes, key=lambda x: x.start)
    # any overlapping notes ?
    for i in range(len(notes)-1):
        if notes[i].end > notes[i+1].start:
            return False
    return True

def extract_melody_notes(midi_path, filter_sf=False, min_notes=5):

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {str(e)}")
        return []
    
    melodies = []

    tempo = pm.get_tempo_changes()[1][0]
    beat_duration = 60.0 / tempo
    
    for instrument in pm.instruments:
        if not instrument.notes:
            continue
        
        notes = sorted(instrument.notes, key=lambda x: x.start)
        
        if is_strictly_monophonic(notes):
            melody, add_melody = get_melody_representation(notes, filter_sf, beat_duration)
            
            if add_melody and len(melody) >= min_notes:
                melodies.append(melody)
    
    return melodies

def infer_key_from_notes(notes):
    s = stream.Stream()
    for n in notes:
        m21_note = note.Note()
        m21_note.pitch.midi = n.pitch
        # set start offset to note.start
        s.insert(n.start, m21_note)
    key = s.analyze('key')
    tonic_pc = key.tonic.pitchClass
    mode = key.mode
    return tonic_pc, mode

def pitch_class_to_scale_degree(pitch_class, tonic_pc, mode):
    if mode == 'major':
        scale = [0, 2, 4, 5, 7, 9, 11]
    elif mode == 'minor':
        scale = [0, 2, 3, 5, 7, 8, 10]
    else: # major if unknown
        scale = [0, 2, 4, 5, 7, 9, 11]

    rel_pc = (pitch_class - tonic_pc) % 12
    if rel_pc in scale:
        scale_degree = scale.index(rel_pc)
    else:
        scale_degree = UNDEFINED

    key_membership = int(rel_pc in scale)
    return scale_degree, key_membership

def get_register(pitch):
    if pitch < 48:
        return 0  # Low register
    elif pitch < 72:
        return 1  # Mid register
    else:
        return 2  # High register

def get_melody_representation(notes, filter_sf, beat_duration):

    if filter_sf:
        for note in notes:
            dur = note.end - note.start
            # sixty-fourth is IDyOM's MAX DURATION
            dur_ratio = dur / beat_duration
            if dur_ratio <= SIXTY_FOURTH_DURATION:
                return None, False

    tonic_pc, mode = infer_key_from_notes(notes)
    first_pitch = notes[0].pitch

    melody = []
    for i, note in enumerate(notes):

        note_representation = dict()

        # PITCH
        pitch = note.pitch
        note_representation['pitch'] = pitch

        # DURATION
        dur = note.end - note.start
        note_representation['duration'] = dur

        # SYMBOLIC_DURATION
        dur_ratio = dur / beat_duration
        symbolic_duration = get_note_type(dur_ratio)
        note_representation['symbolic_duration'] = symbolic_duration
        
        # ONSET
        onset = note.start
        note_representation['onset'] = onset

        # INTERVAL
        if i > 0:
            interval = note.pitch - notes[i-1].pitch
        else:
            interval = UNDEFINED
        note_representation['interval'] = interval

        # CONTOUR
        if i > 0:
            contour = np.sign(note.pitch - notes[i-1].pitch)
        else:
            contour = UNDEFINED
        note_representation['contour'] = contour

        # PITCH CLASS
        pitch_class = note.pitch % 12
        note_representation['pitch_class'] = pitch_class

        # BEAT POSITION
        beat_position = (onset % beat_duration) / beat_duration
        note_representation['beat_position'] = beat_position

        # INTER-ONSET INTERVAL (IOI)
        if i > 0:
            ioi = note.start - notes[i - 1].start
        else:
            ioi = 0
        note_representation['ioi'] = ioi

        # SCALE_DEGREE
        scale_degree, key_membership = pitch_class_to_scale_degree(pitch_class, tonic_pc, mode)
        note_representation['scale_degree'] = scale_degree

        # KEY MEMBERSHIP
        note_representation['key_membership'] = key_membership

        # SELF-SIMILARITY
        window_size = 5  # TODO
        recent_pitches = [notes[j].pitch for j in range(max(0, i - window_size), i)]
        is_repeated_pitch = int(pitch in recent_pitches)
        note_representation['is_repeated_pitch'] = is_repeated_pitch

        # REGISTER
        register = get_register(pitch)
        note_representation['register'] = register

        # PHRASE
        if i == 0 or ioi >= PHRASE_IOI_THRESHOLD * beat_duration:
            phrase = 1  # Start of phrase
        else:
            phrase = 0  # Inside phrase
        note_representation['phrase'] = phrase

        # CPINTFIP
        cpintfip = pitch - first_pitch
        note_representation['cpintfip'] = cpintfip

        melody.append(note_representation)

    return melody, True

def get_feature_encoded_vector(feature, vals):
    feature_vec = None

    if feature == 'pitch':
        feature_vec = torch.zeros(FEATURE_DIM['pitch'], dtype=torch.float32)
        feature_vec[vals[0]] = 1.0

    elif feature == 'duration':
        feature_vec = torch.tensor([min(vals[0] / vals[1], 1.0)], dtype=torch.float32)

    elif feature == 'symbolic_duration':
        feature_vec = torch.zeros(FEATURE_DIM['symbolic_duration'], dtype=torch.float32)
        feature_vec[vals[0]] = 1.0

    elif feature == 'interval':
        feature_vec = torch.zeros(FEATURE_DIM['interval'], dtype=torch.float32)
        if vals[0] != UNDEFINED:
            idx = vals[0] + 36  # shift interval range (-36 to 36) -> (0 to 72)
            if 0 <= idx < FEATURE_DIM['interval']:
                feature_vec[idx] = 1.0

    elif feature == 'onset':
        feature_vec = torch.tensor([min(vals[0] / vals[1], 1.0)], dtype=torch.float32)

    elif feature == 'contour':
        feature_vec = torch.zeros(FEATURE_DIM['contour'], dtype=torch.float32)
        if vals[0] != UNDEFINED:
            idx = vals[0] + 1  # contour values: -1, 0, 1 → indices: 0, 1, 2
            feature_vec[idx] = 1.0

    elif feature == 'pitch_class':
        feature_vec = torch.zeros(12, dtype=torch.float32)
        feature_vec[vals[0]] = 1.0

    elif feature == 'beat_position':
        feature_vec = torch.tensor([vals[0]], dtype=torch.float32)

    elif feature == 'ioi':
        feature_vec = torch.zeros(FEATURE_DIM['ioi'], dtype=torch.float32)
        if vals[0] != UNDEFINED:
            feature_vec = torch.tensor([min(vals[0] / vals[1], 1.0)], dtype=torch.float32)

    elif feature == 'scale_degree':
        feature_vec = torch.zeros(FEATURE_DIM['scale_degree'], dtype=torch.float32)
        if vals[0] != UNDEFINED:
            feature_vec[vals[0]] = 1.0

    elif feature == 'key_membership':
        feature_vec = torch.tensor([vals[0]], dtype=torch.float32)

    elif feature == 'is_repeated_pitch':
        feature_vec = torch.tensor([vals[0]], dtype=torch.float32)

    elif feature == 'register':
        feature_vec = torch.zeros(FEATURE_DIM['register'], dtype=torch.float32)
        feature_vec[vals[0]] = 1.0

    elif feature == 'phrase':
        feature_vec = torch.zeros(FEATURE_DIM['phrase'], dtype=torch.float32)
        if vals[0] == 1:
            feature_vec[1] = 1.0  # phrase start
        else:
            feature_vec[0] = 1.0  # not phrase start

    elif feature == 'cpintfip':
        feature_vec = torch.zeros(FEATURE_DIM['cpintfip'], dtype=torch.float32)
        idx = vals[0] + 36  # shift interval range (-36 to 36) -> (0 to 72)
        if 0 <= idx < FEATURE_DIM['cpintfip']:
            feature_vec[idx] = 1.0

    return feature_vec

def get_note_vec(note_representation, max_duration, max_onset, max_ioi, features, device='cuda'):
    # WARNING: We use multiple if statement instead of a loop over features so that we know the input representation is consistent
    note_vec = []
    feature = 'pitch'
    if feature in features:
        pitch = note_representation[feature]
        pitch_one_hot = get_feature_encoded_vector(feature, [pitch])
        note_vec.append(pitch_one_hot)
    
    feature = 'duration'
    if feature in features:
        duration = note_representation[feature]
        normalized_duration = get_feature_encoded_vector(feature, [duration, max_duration])
        note_vec.append(normalized_duration)

    feature = 'symbolic_duration'
    if feature in features:
        note_type = note_representation[feature]
        note_type_one_hot = get_feature_encoded_vector(feature, [note_type])
        note_vec.append(note_type_one_hot)

    feature = 'interval'
    if feature in features:
        interval = note_representation[feature]
        interval_one_hot = get_feature_encoded_vector(feature, [interval])
        note_vec.append(interval_one_hot) 

    feature = 'onset'
    if feature in features:
        onset = note_representation[feature]
        normalized_onset = get_feature_encoded_vector(feature, [onset, max_onset])
        note_vec.append(normalized_onset)

    feature = 'contour'
    if feature in features:
        contour = note_representation[feature]
        contour_one_hot = get_feature_encoded_vector(feature, [contour])
        note_vec.append(contour_one_hot)

    feature = 'pitch_class'
    if feature in features:
        pitch_class = note_representation[feature]
        pc_one_hot = get_feature_encoded_vector(feature, [pitch_class])
        note_vec.append(pc_one_hot)

    feature = 'beat_position'
    if feature in features:
        beat_pos = note_representation[feature]
        beat_pos_vec = get_feature_encoded_vector(feature, [beat_pos])
        note_vec.append(beat_pos_vec)

    feature = 'ioi'
    if feature in features:
        ioi = note_representation[feature]
        normalized_ioi = get_feature_encoded_vector(feature, [ioi, max_ioi])
        note_vec.append(normalized_ioi)

    feature = 'scale_degree'
    if feature in features:
        scale_degree = note_representation[feature]
        scale_degree_one_hot = get_feature_encoded_vector(feature, [scale_degree])
        note_vec.append(scale_degree_one_hot)

    feature = 'key_membership'
    if feature in features:
        in_key = note_representation[feature]
        key_vec = get_feature_encoded_vector(feature, [in_key])
        note_vec.append(key_vec)

    feature = 'is_repeated_pitch'
    if feature in features:
        is_repeated = note_representation[feature]
        repeated_vec = get_feature_encoded_vector(feature, [is_repeated])
        note_vec.append(repeated_vec)

    feature = 'register'
    if feature in features:
        register = note_representation[feature]
        register_one_hot = get_feature_encoded_vector(feature, [register])
        note_vec.append(register_one_hot)

    feature = 'phrase'
    if feature in features:
        phrase = note_representation[feature]
        phrase_vec = get_feature_encoded_vector(feature, [phrase])
        note_vec.append(phrase_vec)

    if 'cpintfip' in features:
        cpintfip = note_representation['cpintfip']
        cpintfip_one_hot = get_feature_encoded_vector('cpintfip', [cpintfip])
        note_vec.append(cpintfip_one_hot)

    return torch.cat(note_vec)
