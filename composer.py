import numpy as np
import simpleaudio as sa
import random

# For MIDI note output
def freq_to_midi(freq):
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def note_index_to_name(index, base_freq, scale_pattern, octave_offset=0):
    # index: 0-based index in the scale
    # base_freq: base frequency of the scale
    # scale_pattern: list of scale steps (e.g., [0,2,3,5,7,8,10])
    # octave_offset: which octave (0 = base octave)
    # Returns (note_name, octave, midi_number)
    # Find the base note index and octave
    # Find the key index (0-11) for the base note
    # We need to find which NOTE_NAMES matches base_freq best
    # We'll use the closest note in NOTE_NAMES to base_freq
    base_midi = freq_to_midi(base_freq)
    base_note_index = base_midi % 12
    base_octave = base_midi // 12 - 1
    # Now, for the given scale index, compute the semitone offset
    scale_step = scale_pattern[index % len(scale_pattern)]
    octave = base_octave + (index // len(scale_pattern)) + octave_offset
    note_index = (base_note_index + scale_step) % 12
    note_name = NOTE_NAMES[note_index]
    midi_number = (octave + 1) * 12 + note_index
    return note_name, octave, midi_number

# ==========================================================
# GLOBAL CONFIG
# ==========================================================

SAMPLE_RATE = 44100
BPM = 120
BEATS_PER_BAR = 4
BARS = 4
NOTE_DURATION = 60 / BPM
MODE = "minor"  # "major" or "minor"

# ==========================================================
# BASE FREQUENCY SYSTEM (Equal Temperament)
# ==========================================================

USE_RANDOM_BASE = True     # If False → use MANUAL_BASE_FREQ
MANUAL_BASE_FREQ = 220.0   # Used if USE_RANDOM_BASE = False

A4_FREQ = 440.0
A4_INDEX = 9  # A in 0-11 system

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

def note_to_freq(note_index, octave):
    n = note_index - A4_INDEX + (octave - 4) * 12
    return A4_FREQ * (2 ** (n / 12))

def generate_base_frequency():
    if USE_RANDOM_BASE:
        key_index = random.randint(0, 11)
        octave = random.choice([3, 4])
        freq = note_to_freq(key_index, octave)
        print("Random Key Selected:", NOTE_NAMES[key_index], "Octave:", octave)
        return freq
    else:
        print("Manual Base Frequency Used:", MANUAL_BASE_FREQ)
        return MANUAL_BASE_FREQ

def semitone_freq(base, n):
    return base * (2 ** (n / 12))

# ==========================================================
# SCALE SYSTEM
# ==========================================================

SCALES = {
    "major":  [0, 2, 4, 5, 7, 9, 11],
    "minor":  [0, 2, 3, 5, 7, 8, 10]
}

def build_scale(base_freq, pattern, octaves=3):
    freqs = []
    for o in range(octaves):
        for step in pattern:
            semitone = step + 12 * o
            freq = semitone_freq(base_freq, semitone)
            freqs.append(freq)
    return freqs

# ==========================================================
# FUNCTIONAL HARMONY
# ==========================================================

FUNCTIONS = {
    "T":  [0, 5],
    "PD": [1, 3],
    "D":  [4, 6]
}

FUNCTION_FLOW = {
    "T":  ["PD", "D"],
    "PD": ["D"],
    "D":  ["T"]
}

TENSION_MAP = {
    "T": 0,
    "PD": 1,
    "D": 2
}

def generate_progression(length):
    current = "T"
    progression = []
    functions = []

    for _ in range(length):
        degree = random.choice(FUNCTIONS[current])
        progression.append(degree)
        functions.append(current)
        current = random.choice(FUNCTION_FLOW[current])

    progression[-1] = 0
    functions[-1] = "T"

    return progression, functions

# ==========================================================
# CHORD BUILDER
# ==========================================================

def get_triad(scale_degree, scale_freqs, scale_len):
    root = scale_degree
    third = (root + 2) % scale_len
    fifth = (root + 4) % scale_len

    return [
        scale_freqs[root],
        scale_freqs[third],
        scale_freqs[fifth]
    ]

# ==========================================================
# MOTIF SYSTEM
# ==========================================================

def generate_motif(scale_len, length=4):
    motif = []
    current = random.randint(0, scale_len - 1)

    for _ in range(length):
        step = random.choice([-2, -1, 1, 2])
        current = max(0, min(scale_len - 1, current + step))
        motif.append(current)

    return motif

def vary_motif(motif, tension, scale_len):
    variation = []
    shift = random.choice([-1, 0, 1])

    for note in motif:
        if tension == 2:
            note += random.choice([-2, -1, 0, 1, 2])
        note = max(0, min(scale_len - 1, note + shift))
        variation.append(note)

    return variation

# ==========================================================
# MELODY GENERATOR
# ==========================================================

def generate_melody(progression, functions, scale_len):
    melody = []
    motif = generate_motif(scale_len)

    for bar in range(BARS):
        tension = TENSION_MAP[functions[bar]]
        bar_notes = vary_motif(motif, tension, scale_len)

        chord_degree = progression[bar]
        chord_tones = [
            chord_degree,
            (chord_degree + 2) % scale_len,
            (chord_degree + 4) % scale_len
        ]

        # Strong beats = chord tones
        for beat in [0, 2]:
            bar_notes[beat] = random.choice(chord_tones)

        melody.append(bar_notes)

    melody[-1][-1] = 0

    return melody

# ==========================================================
# BASS GENERATOR
# ==========================================================

def generate_bass(progression, scale_len):
    bass = []
    prev = 0

    for degree in progression:
        if abs(degree - prev) > 3:
            if degree > prev:
                degree -= scale_len
            else:
                degree += scale_len
        bass.append(degree)
        prev = degree

    return bass

# ==========================================================
# AUDIO ENGINE
# ==========================================================

def generate_tone(freq, dur, vol=0.3):
    samples = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, samples, False)
    wave = np.sin(2 * np.pi * freq * t)

    attack = int(0.01 * samples)
    release = int(0.05 * samples)
    envelope = np.ones(samples)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)

    return wave * envelope * vol

def render(progression, melody, bass, scale_freqs, scale_len):
    full_audio = []

    for bar in range(BARS):
        chord = get_triad(progression[bar], scale_freqs, scale_len)

        for beat in range(BEATS_PER_BAR):
            note_index = melody[bar][beat]
            note_freq = scale_freqs[note_index]

            m = generate_tone(note_freq, NOTE_DURATION, 0.35)
            c = sum(generate_tone(f, NOTE_DURATION, 0.08) for f in chord)

            bass_degree = bass[bar] % scale_len
            bass_freq = scale_freqs[bass_degree] / 2
            b = generate_tone(bass_freq, NOTE_DURATION, 0.25)

            full_audio.append(m + c + b)

    song = np.concatenate(full_audio)
    audio = (song * 32767).astype(np.int16)
    sa.play_buffer(audio, 1, 2, SAMPLE_RATE).wait_done()

# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=== ADVANCED PROCEDURAL TONAL COMPOSER ===")

    base_freq = generate_base_frequency()

    scale_pattern = SCALES[MODE]
    scale_freqs = build_scale(base_freq, scale_pattern, 3)
    scale_len = len(scale_pattern)

    print("Mode:", MODE)
    print("Base Frequency:", round(base_freq, 2), "Hz")
    print("Bars:", BARS)
    print("Tempo:", BPM)
    print()

    progression, functions = generate_progression(BARS)
    melody = generate_melody(progression, functions, scale_len)
    bass = generate_bass(progression, scale_len)

    print("Functional flow:", functions)
    print("Chord degrees:", progression)
    print("Bass degrees:", bass)
    print("Melody:")
    for i, bar in enumerate(melody):
        print(f" Bar {i+1}:", bar)

    # Output MIDI notes per bar
    print("\nMidi by Bar:")
    for i, bar in enumerate(melody):
        bar_notes = []
        for idx in bar:
            note_name, octave, midi_number = note_index_to_name(idx, base_freq, scale_pattern)
            bar_notes.append(f"{note_name}{octave}({midi_number})")
        print(f" Bar {i+1}: ", ", ".join(bar_notes))

    render(progression, melody, bass, scale_freqs, scale_len)

if __name__ == "__main__":
    main()
