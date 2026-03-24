# Rezonance

A deep learning research playground for monophonic pitch detection and estimation, built on PyTorch. The project explores training convolutional neural networks to classify the pitch of raw audio waveforms, using both synthetic and real musical note datasets.

## Overview

Rezonance trains models to estimate the fundamental frequency (F0) of an audio signal directly from raw waveform buffers. The core approach uses **additive synthesis** to generate diverse synthetic training data ; audio signals are constructed as sums of sinusoidal harmonics at integer multiples of a fundamental frequency, with instrument-specific amplitude and phase distributions. Models are validated against real-world recordings from the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth).

**Default audio parameters:**
- Sample rate: 16,000 Hz
- Buffer size: 1,024 samples (~64ms windows)
- A4 reference: 440 Hz

## Project Structure

```
rezonance/
├── src/rezonance/
│   ├── audioutils/
│   │   ├── waveform_generators.py  # Additive synthesis engine
│   │   ├── noise_generators.py     # Spectral noise generators
│   │   └── pitch_utils.py          # Pitch/frequency conversions
│   ├── datasets/
│   │   ├── synth_dataset.py        # Synthetic dataset generator
│   │   └── real_dataset.py         # NSynth H5/file loaders
│   ├── models/
│   │   ├── convmodel.py            # Primary 1D CNN model
│   │   ├── fcmodel.py              # Fully-connected model
│   │   └── smalltestmodel.py       # Lightweight test model
│   ├── training.py                 # Trainer class with train/validate loop
│   ├── objectives.py               # Training objectives and metrics
│   ├── transforms.py               # Data augmentation transforms
│   ├── defaults.py                 # Global audio constants
│   └── utils.py                    # Device detection utilities
├── scripts/
│   ├── train.py                    # Main training entry point
│   ├── infer.py                    # Run inference on test data
│   ├── store_dataset.py            # Pre-process NSynth to HDF5
│   ├── test_overfit.py             # Sanity-check overfitting test
│   └── experiments/                # Exploratory scripts
├── data/
│   ├── nsynth-test/                # NSynth test split (WAV + metadata)
│   └── nsynth-valid/               # NSynth validation split
├── figures/                        # Training plots (loss, accuracy)
├── saved_models/                   # Persisted model weights (.pth)
├── docs/                           # Reference papers (CREPE, PESTO)
└── PLAN.md                         # Development roadmap
```

## Installation

Requires Python 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
uv pip install -e .
```

GPU support (CUDA 13.0) is automatically configured on Linux/Windows via the PyTorch index. On macOS, CPU or MPS is used automatically.

## Usage

### Training

```bash
python scripts/train.py
```

Trains a `ConvModel` on a mixed synthetic dataset (multiple instrument timbres, data augmentations) and validates against a pre-processed NSynth HDF5 file. The model is saved to `saved_models/model.pth` on completion or keyboard interrupt. Training and validation curves are saved to `figures/`.

### Preparing the validation dataset

```bash
python scripts/store_dataset.py
```

Pre-processes the NSynth validation split into an HDF5 file for faster loading during training.

### Inference

```bash
python scripts/infer.py
```

Loads a saved model and runs inference on random samples from the NSynth test set, displaying the signal waveform alongside true and predicted pitch.

### Overfitting sanity check

```bash
python scripts/test_overfit.py
```

Fits the model on a single batch to verify the training pipeline is working before running a full experiment.

## Core Components

### Synthetic Data Generation (`audioutils/waveform_generators.py`)

`InstrumentSynth` generates audio buffers using additive synthesis. Each signal is constructed as a sum of harmonics at integer multiples of a fundamental frequency, with amplitude (power) and phase drawn from instrument-specific distributions.

Built-in instrument timbres via the `Instrument` factory:

| Instrument | Power Distribution | Notes |
|---|---|---|
| `Instrument.saw` | `1/n` decay | Classic sawtooth |
| `Instrument.square` | `1/n` on odd harmonics | Square wave |
| `Instrument.triangle` | `1/n²` alternating | Triangle wave |
| `Instrument.sine` | Fundamental only | Pure sine |
| `Instrument.random(alpha)` | Random decay at rate `1/n^alpha` | Varied timbre |

### Noise Generation (`audioutils/noise_generators.py`)

`NoiseSynth` generates colored noise via FFT filtering. Multiple noise sources can be combined with `+`.

| Type | Spectral shape |
|---|---|
| `Noise.white(power)` | Flat |
| `Noise.pink(power)` | `1/√f` |
| `Noise.brown(power)` | `1/f` |
| `Noise.blue(power)` | `√f` |
| `Noise.violet(power)` | `f` |

```python
noise = Noise.brown(0.1) + Noise.violet(0.02)
```

### Data Augmentation (`transforms.py`)

Transforms are composable callables applied to waveform tensors at dataset `__getitem__` time:

```python
transform = transforms.random_choice(
    transforms.none(),
    transforms.noise(Noise.brown(0.1) + Noise.violet(0.02)),
    transforms.compose(
        transforms.noise(Noise.white(0.05)),
        transforms.scaling(1, 0.8, BUFFER_SIZE),
    ),
    transforms.mask(50, 0),
)
```

| Transform | Description |
|---|---|
| `noise(synth)` | Add colored noise, re-normalize by std |
| `mask(size, value)` | Zero out a random time window |
| `scaling(low, high, size)` | Apply a linear gain envelope |
| `compose(*transforms)` | Sequential pipeline |
| `random_choice(*transforms)` | Uniform random selection |
| `none()` | Identity (no-op) |

### Training Objectives (`objectives.py`)

The `Objective` interface decouples label encoding, loss computation, and metrics from the model and trainer, making it easy to switch between formulations.

| Objective | Output | Loss | Use case |
|---|---|---|---|
| `BasicObjective` | Pitch number (scalar) | MSE | Direct regression |
| `CyclicPitchObjective` | `(sin, cos[, octave])` | MSE | Pitch-class aware regression |
| `NoteClassifierObjective(min, max, step)` | One-hot class logits | CrossEntropy | Classification |

### Model (`models/convmodel.py`)

`ConvModel` is a 1D CNN that operates on raw waveform buffers:

- **Input**: `(batch, buffer_size)` ; raw audio samples
- **Architecture**: 6 convolutional blocks (Conv1d → ReLU → MaxPool1d → BatchNorm → Dropout 0.25), followed by a linear output layer
- **Output**: `(batch, output_size)` ; depends on the chosen objective

```
Input (1024,)
  → Conv1d(1→1024, k=512, s=4) → Pool → BN
  → Conv1d(1024→128, k=64)     → Pool → BN
  → Conv1d(128→128, k=64)      → Pool → BN   ×2
  → Conv1d(128→256, k=64)      → Pool → BN
  → Conv1d(256→512, k=64)      → Pool → BN
  → Flatten → Linear(2048→output_size)
```

### Trainer (`training.py`)

```python
trainer = Trainer(model, optimizer, objective)
trainer.train(
    train_dataset,
    validation_dataset,
    nb_epoch=100,
    batch_size=512,
    validate_every=1,
    log_epochs=1,
)
```

Training and validation metrics are stored in `trainer.train_history` / `trainer.validation_history` as lists of `Metric` objects.

## Datasets

### Synthetic (`datasets/synth_dataset.py`)

`InstrumentSynthDataset` generates signals on-the-fly at construction time and stores them in memory. Signals are unit-normalized by standard deviation.

```python
dataset = InstrumentSynthDataset(
    pitch_step=1/4,       # quarter-tone resolution
    per_pitch=250,        # samples per pitch
    transform=transform,
    instrument=Instrument.random(1.5, ...),
    sample_rate=16_000,
    min_pitch=36,
    max_pitch=84,
)
```

### Real (`datasets/real_dataset.py`)

- **`H5Dataset`**: Loads pre-processed data from an HDF5 file (fast, suitable for validation loops).
- **`FileDataset`**: Loads directly from NSynth WAV files and `examples.json` metadata, extracting `element_per_file` non-overlapping buffer windows per audio file.

## Dependencies

| Package | Version |
|---|---|
| Python | 3.14.x |
| PyTorch | 2.10.x |
| torchaudio | 2.10.x |
| NumPy | 2.3.x |
| scikit-learn | 1.8.x |
| pandas | 3.0.x |
| h5py | latest |
| matplotlib | latest |
| sounddevice | latest |

## Roadmap

See [`PLAN.md`](PLAN.md) for the current development status. Planned work includes polyphonic detection and comparison against classical pitch estimation methods (pYIN).
