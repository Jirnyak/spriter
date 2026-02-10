import numpy as np
import simpleaudio as sa
import time
import random
import math

# ==============================
# GLOBAL CONFIG
# ==============================

BASE_FREQ = 220.0      # lowest note frequency
BPM = 120
BLOCK_SIZE = 8

NOTE_DURATION = 60 / BPM
SAMPLE_RATE = 44100

VOL_MAIN = 0.35
VOL_SUPPORT = 0.22
VOL_SHADOW = 0.15

# ==============================
# SCALE (RELATIVE, ORDERED)
# ==============================

# Extended natural minor: lower octave + main octave
# Negative indices for bass notes below BASE_FREQ
SCALE_STEPS = [-12, -10, -8, -7, -5, -3, -2, 0, 2, 3, 5, 7, 8, 10, 12, 14]
SCALE_SIZE = len(SCALE_STEPS)

# Index roles (adjusted for extended range)
BASS = [0, 1, 2, 3]      # Lower octave
LOW = [4, 5, 6, 7, 8]     # Lower middle
MID = [7, 8, 9, 10]       # Middle
HIGH = [10, 11, 12, 13, 14, 15]  # Upper

HOME = [7, 9]  # Tonic and third (now at index 7, 9)
BASS_HOME = [0, 2]  # Bass tonic notes

# Gravity weights (extended)
WEIGHTS = {
    0: 4,   # Bass notes
    1: 1,
    2: 3,
    3: 1,
    4: 2,
    5: 1,
    6: 2,
    7: 5,   # Main tonic
    8: 2,
    9: 4,   # Main third
    10: 2,
    11: 3,  # Fifth
    12: 2,
    13: 2,
    14: 1,
    15: 1,
}

# ==============================
# RHYTHM
# ==============================

RHYTHMS = [
    ["-", "n", "n", "-", "n", "-", "n", "-"],  # Classic syncopation
    ["n", "-", "n", "-", "n", "n", "-", "-"],  # Steady pulse
    ["-", "n", "-", "n", "n", "-", "n", "-"],  # Offbeat
    ["n", "n", "-", "n", "-", "-", "n", "n"],  # Quick doubles
    ["-", "-", "n", "n", "-", "n", "n", "-"],  # Delayed start
    ["n", "-", "-", "n", "n", "-", "n", "-"],  # Sparse
    ["n", "-", "n", "n", "-", "n", "-", "n"],  # Dense RPG
    ["-", "n", "n", "n", "-", "-", "n", "-"],  # Burst pattern
]

# Bass rhythm patterns (for support voice)
BASS_RHYTHMS = [
    ["n", "-", "-", "-", "n", "-", "-", "-"],  # 4/4 kick pattern
    ["n", "-", "n", "-", "n", "-", "n", "-"],  # Walking bass
    ["n", "-", "-", "n", "-", "-", "n", "-"],  # Sparse foundation
    ["n", "n", "-", "-", "n", "-", "-", "-"],  # Double kick
]

# ==============================
# AUDIO
# ==============================

def freq_from_index(idx):
    semitones = SCALE_STEPS[idx]
    return BASE_FREQ * (2 ** (semitones / 12))

def generate_tone(freq, dur, vol):
    """Generate a tone waveform with ADSR envelope."""
    samples = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, samples, False)
    
    # Sine wave
    wave = np.sin(2 * np.pi * freq * t)
    
    # ADSR envelope
    attack = int(samples * 0.05)  # 5% attack
    decay = int(samples * 0.1)    # 10% decay
    release = int(samples * 0.2)  # 20% release
    sustain_level = 0.7
    
    envelope = np.ones(samples)
    
    # Attack
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    
    # Decay
    if decay > 0 and attack + decay < samples:
        envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
    
    # Sustain (already set to value between decay and release)
    sustain_start = attack + decay
    sustain_end = max(sustain_start, samples - release)
    if sustain_end > sustain_start:
        envelope[sustain_start:sustain_end] = sustain_level
    
    # Release
    if release > 0:
        envelope[-release:] = np.linspace(sustain_level, 0, release)
    
    return wave * envelope * vol

def play_audio(audio_data):
    """Play audio data."""
    audio = (audio_data * 32767).astype(np.int16)
    sa.play_buffer(audio, 1, 2, SAMPLE_RATE).wait_done()

# ==============================
# GENERATION CORE
# ==============================

def weighted_pick(indices):
    return random.choices(indices, weights=[WEIGHTS[i] for i in indices], k=1)[0]

def step_motion(idx, allow_jump=True, range_limit=None):
    """Get candidate notes following step-wise motion rules."""
    opts = []
    
    # Stepwise motion (preferred)
    if idx > 0:
        opts.append(idx - 1)
    if idx < SCALE_SIZE - 1:
        opts.append(idx + 1)
    
    # Occasional small jumps (thirds)
    if allow_jump and random.random() < 0.3:
        if idx > 1:
            opts.append(idx - 2)
        if idx < SCALE_SIZE - 2:
            opts.append(idx + 2)
    
    # Filter by range if specified
    if range_limit:
        opts = [o for o in opts if o in range_limit]
    
    return opts or [idx]

def generate_main(note_count, arc="wave"):
    """Generate main melody with musical phrasing."""
    melody = []
    current = random.choice(HOME)
    melody.append(current)
    
    peak_reached = False
    
    for i in range(note_count - 2):
        progress = i / max(1, note_count - 2)
        
        candidates = step_motion(current, allow_jump=(i % 2 == 0))
        
        if arc == "rising" and not peak_reached:
            # Favor upward motion toward high notes
            upward = [c for c in candidates if c > current]
            if upward and current < 11:
                candidates = upward + MID + [h for h in HIGH if h <= 13]
            if current >= 11:
                peak_reached = True
                
        elif arc == "falling":
            # Favor downward motion toward low notes
            downward = [c for c in candidates if c < current]
            if downward:
                candidates = downward + LOW
            else:
                candidates += LOW
                
        elif arc == "wave":
            # Wave: rise then fall
            if progress < 0.5 and current < 10:
                # Rising phase
                candidates += MID + [h for h in HIGH if h <= 12]
            elif progress >= 0.5:
                # Falling phase
                candidates += LOW + [h for h in HOME]
            else:
                candidates += MID
        
        # Add some randomness for variety
        if random.random() < 0.2:
            candidates += HOME
        
        current = weighted_pick(list(set(candidates)))
        melody.append(current)
    
    # End on home note (tonic)
    melody.append(random.choice(HOME))
    return melody

# ==============================
# PARALLEL VOICES (INHERITED)
# ==============================

def generate_support(main, use_bass_rhythm=True):
    """Generate bass/harmonic support with proper intervals."""
    support = []
    bass_rhythm = random.choice(BASS_RHYTHMS) if use_bass_rhythm else None
    
    for i, note in enumerate(main):
        if note == "-":
            support.append("-")
            continue
        
        # Apply bass rhythm pattern - more sparse
        if bass_rhythm and bass_rhythm[i % len(bass_rhythm)] == "-" and random.random() < 0.6:
            support.append("-")
            continue
        
        # Harmonize with bass notes for fuller sound
        # Map high notes to bass range
        if note in HOME:  # Tonic notes
            # Use bass tonic or fifth
            support.append(random.choice(BASS_HOME + [4]))  # Bass tonic or low fifth
        elif note in MID:  # Mid notes
            # Use bass tonic or third
            support.append(random.choice(BASS_HOME + [2]))
        elif note in HIGH:  # High notes
            # Use lower harmonies (bass or low range)
            support.append(random.choice(BASS_HOME + LOW[:2]))
        elif note in LOW:
            # Already low, use bass notes
            support.append(random.choice(BASS_HOME))
        else:
            # Default to bass tonic
            support.append(random.choice(BASS_HOME))
    
    return support

def generate_shadow(main):
    """Generate rhythmic counterpoint, arpeggios, and echoes."""
    shadow = []
    delay_buffer = ["-", "-"]
    last_melody_note = None
    
    for i, n in enumerate(main):
        if n != "-":
            last_melody_note = n
        
        # Create varied rhythmic patterns
        if i % 3 == 0 and last_melody_note is not None and random.random() < 0.5:
            # Arpeggio: play interval above or below
            if last_melody_note in HIGH:
                # High notes - add lower harmony
                offset = random.choice([-2, -3, -4])  # Third or fifth down
            elif last_melody_note in BASS:
                # Bass notes - add upper harmony
                offset = random.choice([2, 3, 4])  # Third or fifth up
            else:
                # Mid notes - vary direction
                offset = random.choice([-2, -3, 2, 3])
            
            target = last_melody_note + offset
            if 0 <= target < SCALE_SIZE:
                shadow.append(target)
            else:
                shadow.append("-")
                
        elif i % 2 == 1 and random.random() < 0.5:
            # Echo delayed note
            prev_note = delay_buffer[-1]
            if prev_note != "-":
                shadow.append(prev_note)
            else:
                shadow.append("-")
        else:
            # Rest for space
            shadow.append("-")
        
        delay_buffer.append(n)
    
    return shadow

def apply_rhythm(seq, rhythm):
    out, i = [], 0
    for r in rhythm:
        if r == "-":
            out.append("-")
        else:
            out.append(seq[i] if i < len(seq) else "-")
            i += 1
    return out

# ==============================
# BLOCK
# ==============================

def generate_block():
    rhythm = random.choice(RHYTHMS)
    arc = random.choice(["wave", "rising", "falling", "wave", "rising"])  # Favor wave and rising
    note_count = rhythm.count("n")

    main = generate_main(note_count, arc)
    support = generate_support(main, use_bass_rhythm=True)
    shadow = generate_shadow(main)

    return (
        apply_rhythm(main, rhythm),
        apply_rhythm(support, rhythm),
        apply_rhythm(shadow, rhythm),
        arc
    )

# ==============================
# PLAYBACK
# ==============================

def play_block(main, support, shadow):
    """Play all voices synchronously by mixing waveforms."""
    samples_per_step = int(SAMPLE_RATE * NOTE_DURATION)
    total_samples = samples_per_step * len(main)
    mixed_audio = np.zeros(total_samples)
    
    for i, (m, s, sh) in enumerate(zip(main, support, shadow)):
        start = i * samples_per_step
        end = start + samples_per_step
        
        # Mix main voice
        if m != "-":
            mixed_audio[start:end] += generate_tone(freq_from_index(m), NOTE_DURATION, VOL_MAIN)
        
        # Mix support voice
        if s != "-":
            mixed_audio[start:end] += generate_tone(freq_from_index(s), NOTE_DURATION, VOL_SUPPORT)
        
        # Mix shadow voice
        if sh != "-":
            mixed_audio[start:end] += generate_tone(freq_from_index(sh), NOTE_DURATION, VOL_SHADOW)
    
    # Play the mixed audio
    play_audio(mixed_audio)

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    print("=== UNIVERSAL PROCEDURAL MUSIC ENGINE ===")
    print(f"Base freq: {BASE_FREQ} Hz")

    blocks = int(input("Blocks (default 4): ") or "4")

    for i in range(blocks):
        main, support, shadow, arc = generate_block()
        print(f"\nBlock {i+1} | arc={arc}")
        print("Main:   ", main)
        print("Support:", support)
        print("Shadow: ", shadow)
        play_block(main, support, shadow)
