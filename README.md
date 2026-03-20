# Formative 2: Multimodal Data Preprocessing
### Group 4 — African Leadership University

**User Identity and Product Recommendation System**
A multimodal biometric authentication pipeline that verifies a user's identity through facial recognition and voice verification before delivering personalised product recommendations.

---

## Team

| Name | Role |
|------|------|
| Glory Uwase | Experiment 1 — Pretrained Embedding Approach |
| Kevin Murehwa | Experiment 1 — Pretrained Embedding Approach |
| Justine Uwase | Experiment 2 — Supervised Classification Approach |
| Edwin Bayingana | Experiment 2 — Supervised Classification Approach |

---

## Repository Structure

```
Formtive2-Group4-Multimodal_Data_Preprocessing/
├── experiment_1/               ← Glory & Kevin: cosine similarity on pretrained embeddings
│   ├── data/
│   │   ├── images/             ← Per-person face photos
│   │   └── audio/              ← Per-person voice recordings
│   ├── data/processed/         ← Generated feature CSVs and merged dataset
│   ├── data/raw/               ← Original tabular datasets
│   ├── models/                 ← Saved XGBoost model and column reference
│   ├── notebooks/
│   │   └── multimodal_pipeline.ipynb
│   ├── scripts/
│   │   ├── feature_utils.py    ← Extract and save embeddings to CSV
│   │   ├── face_verifier.py    ← Cosine similarity face recognition
│   │   ├── voice_verifier.py   ← Cosine similarity voice verification
│   │   └── auth_gate.py        ← Interactive CLI (run this)
│   └── requirements.txt
│
├── experiment_2/               ← Edwin & Justine: supervised Random Forest classifiers
│   ├── data/
│   │   ├── images/             ← Per-person face photos (includes HEIC-converted files)
│   │   ├── audio/              ← Per-person voice recordings
│   │   └── dataset/            ← Tabular datasets and merged CSV
│   ├── features/               ← Generated feature CSVs and EDA plots
│   ├── models/                 ← Saved model bundles (.pkl)
│   ├── scripts/
│   │   └── app.py              ← Batch demo CLI (run this)
│   ├── formative2_multimodal.ipynb
│   └── requirements.txt
│
└── README.md
```

---

## The Two Approaches

This project was implemented twice using different authentication strategies, allowing a direct comparison of the two techniques on the same dataset.

### Experiment 1 — Pretrained Embedding + Cosine Similarity (Glory & Kevin)

Face images are passed through a pretrained ResNet-18 network (PyTorch / torchvision) with the classification head removed, producing a 512-dimensional embedding per image. Audio clips are processed into a 42-dimensional vector of 40 MFCCs, spectral rolloff, and RMS energy. All vectors are L2-normalised.

At enrolment, embeddings for each person are averaged into a single reference vector. At authentication time, a new input is compared against all stored references using the dot product (cosine similarity). If the best match exceeds the threshold (default 0.80 for both modalities) the user is accepted.

**No model training is required.** Adding a new person only requires re-running the feature extraction script.

### Experiment 2 — Supervised Random Forest Classifiers (Edwin & Justine)

Face features are extracted as a 55-dimensional vector of per-channel colour histograms, channel statistics, and Canny edge density. Audio features are extracted as a 32-dimensional vector combining MFCC mean and standard deviation, spectral rolloff, RMS energy, zero-crossing rate, and spectral centroid.

Six augmented variants are generated per image (rotations, flip, grayscale, brightness changes) and per audio clip (pitch shifts, time stretching, noise addition, reversal), expanding the training set from ~22 images and 14 audio clips to 84 and 56 samples respectively.

Three separate Random Forest classifiers are trained: one for face recognition, one for voice verification, and one for product recommendation. Each outputs a class probability. If the top-class confidence exceeds 0.50, the input is accepted.

---

## Authentication Pipeline

Both experiments implement the same three-step flow:

```
[Face Image]  ──►  Step 1: Facial Recognition  ──► Pass / Deny
                              │ Pass
                              ▼
[Voice Audio] ──►  Step 2: Voice Verification   ──► Pass / Deny
                              │ Pass
                              ▼
               Step 3: Product Recommendation   ──► Show recommendation
```

A user who fails at Step 1 is denied without reaching Step 2. A user who passes Step 1 but fails Step 2 (or whose voice does not match their face identity) is also denied.

---

## Quick Start

### Experiment 1

```bash
cd experiment_1
pip install -r requirements.txt

# 1. Extract embeddings (run once)
python scripts/feature_utils.py data/

# 2. Run the interactive authentication gate
python scripts/auth_gate.py

# Optional: adjust similarity thresholds
python scripts/auth_gate.py --face-thresh 0.75 --voice-thresh 0.75
```

> **Requires ffmpeg** for M4A audio loading and **webcam/microphone** for live capture mode.

### Experiment 2

```bash
cd experiment_2
pip install -r requirements.txt

# 1. Run the Jupyter notebook to train models
jupyter notebook formative2_multimodal.ipynb
# Execute all cells — this generates the .pkl model files in models/

# 2. Authenticate all 4 team members
python scripts/app.py --demo team

# 3. Run the 3 security scenarios
python scripts/app.py --demo all

# 4. Test a specific file
python scripts/app.py --face data/images/Edwin/IMG_5552.jpg \
                       --voice "data/audio/Edwin/Yes, approve.m4a"
```

---

## Demo Scenarios (Experiment 2)

| Flag | What it shows |
|------|---------------|
| `--demo team` | All 4 members authenticate with their real photo and voice, followed by a stranger being denied |
| `--demo all` | Three scenarios: authorised user, unknown face denied, identity mismatch denied |
| `--demo authorized` | Edwin authenticates successfully and receives a recommendation |
| `--demo unauthorized` | Random noise image is rejected at the face step |
| `--demo mismatch` | Edwin's face paired with Justine's voice is rejected at the voice step |

---

## Approach Comparison

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Face features | 512-dim ResNet-18 embedding | 55-dim colour histogram + edge density |
| Voice features | 42-dim MFCC + rolloff + energy | 32-dim MFCC (mean+std) + spectral features |
| Matching | Cosine similarity | Random Forest (probability output) |
| Training needed | No | Yes |
| Threshold | Similarity score (0.80) | Class confidence (0.50) |
| Enrol new user | Add to CSV, no retraining | Retrain all three models |
| Deep learning | PyTorch + torchvision | scikit-learn only |
| Product model | XGBoost | Random Forest |

---

## Demo Video

[https://youtu.be/tv-BkUUN0uk](https://youtu.be/tv-BkUUN0uk)

---

## Notes

- Kevin's images were originally in HEIC format mislabelled as `.jpg`. They were converted using ImageMagick (`convert input.jpg output_fixed.jpg`) before feature extraction. The converted files are the `_fixed.jpg` variants in `experiment_2/data/images/Kevin/`.
- All audio files are in M4A/AAC format. librosa loads them via the ffmpeg backend. Ensure ffmpeg is installed on your system (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Ubuntu).
- The tabular dataset join required extracting the numeric portion from the alphanumeric customer ID format (e.g. `A178` → `178`) before merging the social profiles and transactions tables.
