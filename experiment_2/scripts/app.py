#!/usr/bin/env python3
"""
app.py — Multimodal Authentication & Product Recommendation CLI
=================================================================
Usage:
    python app.py --face <path_to_image> --voice <path_to_audio>
    python app.py --demo authorized
    python app.py --demo unauthorized
    python app.py --demo mismatch

This script simulates the full User Identity and Product Recommendation
System Flow:

  [Face Image] → Facial Recognition ──►  ✓ / ✗
       ↓ (if ✓)
  [Voice Audio] → Voiceprint Verification ──► ✓ / ✗
       ↓ (if ✓)
  [Customer Profile] → Product Recommendation ──► Display

Authors: Glory, Kevin, Justine, Edwin
"""

import os
import sys
import argparse
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
MODEL_DIR   = os.path.join(BASE_DIR, "models")
IMG_DIR     = os.path.join(BASE_DIR, "data", "images")
AUDIO_DIR   = os.path.join(BASE_DIR, "data", "audio")
FEAT_DIR    = os.path.join(BASE_DIR, "features")

CONFIDENCE_THRESHOLD = 0.50
MEMBERS   = ["Glory", "Kevin", "Justine", "Edwin"]
PHRASES   = ["approve", "confirm"]
SR        = 22050

# ── Lazy imports (warn clearly if missing) ────────────────────────────────────
def _import_check():
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("        Run: pip install " + " ".join(missing))
        sys.exit(1)

_import_check()

import cv2
import librosa
import soundfile as sf

# ── Load saved models ─────────────────────────────────────────────────────────
def load_models():
    face_path    = os.path.join(MODEL_DIR, "face_model.pkl")
    voice_path   = os.path.join(MODEL_DIR, "voice_model.pkl")
    product_path = os.path.join(MODEL_DIR, "product_model.pkl")

    for p, name in [(face_path, "face"), (voice_path, "voice"), (product_path, "product")]:
        if not os.path.exists(p):
            print(f"[ERROR] {name} model not found at {p}")
            print("        Please run the Jupyter notebook first to train and save the models.")
            sys.exit(1)

    with open(face_path,    'rb') as f: face_bundle    = pickle.load(f)
    with open(voice_path,   'rb') as f: voice_bundle   = pickle.load(f)
    with open(product_path, 'rb') as f: product_bundle = pickle.load(f)
    return face_bundle, voice_bundle, product_bundle


# ── Feature extraction (must match notebook) ──────────────────────────────────
def extract_image_features(img):
    """Extract 55-dim color histogram + edge features from RGB image."""
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        feats.extend(hist.tolist())
    for ch in range(3):
        feats.append(float(img[:, :, ch].mean()))
        feats.append(float(img[:, :, ch].std()))
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    feats.append(float(edges.mean()))
    return np.array(feats)


def extract_audio_features(signal, sr):
    """Extract 32-dim MFCC + spectral + energy features."""
    if len(signal) < 512:
        signal = np.pad(signal, (0, 512 - len(signal)))
    mfccs    = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    rolloff  = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    rms      = librosa.feature.rms(y=signal)
    zcr      = librosa.feature.zero_crossing_rate(signal)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    feats = []
    for i in range(13):
        feats.append(float(mfccs[i].mean()))
        feats.append(float(mfccs[i].std()))
    feats += [float(rolloff.mean()), float(rolloff.std()),
              float(rms.mean()),     float(rms.std()),
              float(zcr.mean()),     float(centroid.mean())]
    return np.array(feats)


# ── Placeholder generators (for --demo mode) ─────────────────────────────────
def _placeholder_face(expression='neutral', member_idx=0, size=224):
    img = np.ones((size, size, 3), dtype=np.uint8) * 220
    bg_colors = [(220,235,255),(255,235,220),(235,255,235),(250,220,255)]
    img[:] = bg_colors[member_idx % 4]
    cx, cy = size//2, size//2
    skins = [(210,175,130),(240,200,165),(165,110,80),(200,155,115)]
    skin  = skins[member_idx % 4]
    cv2.ellipse(img, (cx, cy), (70, 88), 0, 0, 360, skin, -1)
    hair_cols = [(60,40,20),(20,20,15),(180,145,100),(80,55,40)]
    hair = hair_cols[member_idx % 4]
    cv2.ellipse(img, (cx, cy-10), (70, 88), 0, 180, 360, hair, -1)
    cv2.rectangle(img, (cx-70, cy-100), (cx+70, cy-10), hair, -1)
    lx, rx, ey = cx-28, cx+28, cy-18
    cv2.circle(img, (lx, ey), 9, (255,255,255), -1)
    cv2.circle(img, (rx, ey), 9, (255,255,255), -1)
    cv2.circle(img, (lx, ey), 5, (50,40,35), -1)
    cv2.circle(img, (rx, ey), 5, (50,40,35), -1)
    lip = (int(skin[0]*0.7), int(skin[1]*0.5), int(skin[2]*0.5))
    if expression == 'neutral':
        cv2.line(img, (cx-18, cy+30), (cx+18, cy+30), lip, 3)
    elif expression == 'smiling':
        pts = np.array([[cx-20,cy+28],[cx,cy+38],[cx+20,cy+28]], dtype=np.int32)
        cv2.polylines(img, [pts], False, lip, 3)
    elif expression == 'surprised':
        cv2.ellipse(img, (cx, cy+32), (12, 16), 0, 0, 360, lip, -1)
    return img


def _placeholder_audio(phrase_idx=0, member_idx=0, sr=SR, duration=2.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    base_freqs = [145, 165, 180, 130]
    f0 = base_freqs[member_idx % 4] * (1.0 + 0.1 * phrase_idx)
    audio = np.zeros(len(t), dtype=np.float32)
    for k in range(1, 10):
        audio += (1.0 / k) * np.exp(-0.1 * k) * np.sin(2 * np.pi * f0 * k * t).astype(np.float32)
    n = len(t)
    env = np.ones(n, dtype=np.float32)
    env[:int(0.08*n)] = np.linspace(0, 1, int(0.08*n))
    env[int(0.75*n):] = np.linspace(1, 0, n - int(0.75*n))
    audio = audio * env
    audio += np.random.normal(0, 0.015, len(audio)).astype(np.float32)
    peak = np.max(np.abs(audio))
    return (audio / peak * 0.88) if peak > 0 else audio


# ── Core pipeline functions ───────────────────────────────────────────────────
def recognize_face(img_rgb, face_bundle):
    feat  = extract_image_features(img_rgb).reshape(1, -1)
    feat_s = face_bundle['scaler'].transform(feat)
    name  = face_bundle['encoder'].inverse_transform(
                face_bundle['model'].predict(feat_s))[0]
    conf  = float(face_bundle['model'].predict_proba(feat_s).max())
    return name, conf


def verify_voice(signal, sr, voice_bundle):
    feat   = extract_audio_features(signal, sr).reshape(1, -1)
    feat_s = voice_bundle['scaler'].transform(feat)
    name   = voice_bundle['encoder'].inverse_transform(
                 voice_bundle['model'].predict(feat_s))[0]
    conf   = float(voice_bundle['model'].predict_proba(feat_s).max())
    return name, conf


def recommend_product(member_name, product_bundle):
    feat_cols = product_bundle['features']
    np.random.seed(hash(member_name) % (2**32))
    profile = {
        'engagement_score':        np.random.randint(60, 99),
        'purchase_interest_score': round(np.random.uniform(0.4, 1.0), 2),
        'purchase_amount':         np.random.randint(100, 490),
        'customer_rating':         round(np.random.uniform(2.5, 5.0), 1),
        'sentiment_score':         np.random.choice([0, 1, 2]),
        'purchase_month':          np.random.randint(1, 13),
        'purchase_dow':            np.random.randint(0, 7),
        'purchase_quarter':        np.random.randint(1, 5),
    }
    profile['engagement_x_interest'] = profile['engagement_score'] * profile['purchase_interest_score']
    profile['amount_per_rating']      = profile['purchase_amount'] / max(profile['customer_rating'], 0.1)
    for plat in ['Facebook','Instagram','LinkedIn','TikTok','Twitter']:
        profile[f'platform_{plat}'] = int(np.random.choice([0,1], p=[0.8,0.2]))
    feat_vec = np.array([profile.get(c, 0) for c in feat_cols]).reshape(1, -1)
    feat_s   = product_bundle['scaler'].transform(feat_vec)
    product  = product_bundle['encoder'].inverse_transform(
                   product_bundle['model'].predict(feat_s))[0]
    probas   = dict(zip(product_bundle['encoder'].classes_,
                        product_bundle['model'].predict_proba(feat_s)[0].round(3)))
    return product, probas


# ── Main pipeline ─────────────────────────────────────────────────────────────
SEP = "═" * 65

def run_pipeline(face_img_rgb, voice_signal, voice_sr,
                 face_bundle, voice_bundle, product_bundle, label=""):
    print(f"\n{SEP}")
    if label:
        print(f"  SCENARIO : {label}")
    print(SEP)

    # Step 1 – Face recognition
    print("\n  [STEP 1]  Facial Recognition")
    print("  " + "─"*43)
    face_name, face_conf = recognize_face(face_img_rgb, face_bundle)
    print(f"  Identified as : {face_name}")
    print(f"  Confidence    : {face_conf:.1%}")

    if face_conf < CONFIDENCE_THRESHOLD:
        print(f"\n  ❌  ACCESS DENIED")
        print(f"  Reason : Face confidence {face_conf:.1%} is below threshold ({CONFIDENCE_THRESHOLD:.0%})")
        print(SEP)
        return False

    print("  ✓  Face accepted — proceeding to voice verification")

    # Step 2 – Voice verification
    print("\n  [STEP 2]  Voice Verification")
    print("  " + "─"*43)
    voice_name, voice_conf = verify_voice(voice_signal, voice_sr, voice_bundle)
    print(f"  Voice matched : {voice_name}")
    print(f"  Confidence    : {voice_conf:.1%}")

    if voice_conf < CONFIDENCE_THRESHOLD:
        print(f"\n  ❌  ACCESS DENIED")
        print(f"  Reason : Voice confidence {voice_conf:.1%} is below threshold ({CONFIDENCE_THRESHOLD:.0%})")
        print(SEP)
        return False

    if voice_name != face_name:
        print(f"\n  ❌  ACCESS DENIED")
        print(f"  Reason : Voice identity '{voice_name}' does not match face identity '{face_name}'")
        print(SEP)
        return False

    print("  ✓  Voice verified and matches face identity")

    # Step 3 – Product recommendation
    print("\n  [STEP 3]  Product Recommendation")
    print("  " + "─"*43)
    product, probas = recommend_product(face_name, product_bundle)
    print(f"  Welcome, {face_name}!")
    print(f"  Recommended   : {product.upper()}")
    print(f"\n  Category Probabilities:")
    for cat, p in sorted(probas.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 35)
        print(f"    {cat:<15} {bar:<35} {p:.1%}")

    print(f"\n  ✅  TRANSACTION APPROVED — {face_name} is cleared to purchase {product}")
    print(SEP)
    return True


# ── Argument parsing & entry point ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Authentication & Product Recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --face images/Edwin/Edwin_neutral.jpg --voice audio/Edwin/Edwin_approve.wav
  python app.py --demo authorized
  python app.py --demo unauthorized
  python app.py --demo mismatch
        """
    )
    parser.add_argument('--face',  type=str, help="Path to face image (.jpg/.png)")
    parser.add_argument('--voice', type=str, help="Path to voice recording (.wav/.mp3)")
    parser.add_argument('--demo',  type=str,
                        choices=['authorized','unauthorized','mismatch','all','team'],
                        help="Run a built-in demo scenario. 'team' authenticates all 4 members.")
    args = parser.parse_args()

    if not args.face and not args.demo:
        parser.print_help()
        sys.exit(0)

    print("\n  Loading models…", end=" ")
    face_bundle, voice_bundle, product_bundle = load_models()
    print("done ✓")

    if args.demo:
        scenarios = ['authorized','unauthorized','mismatch'] if args.demo == 'all' else [args.demo]

        # ── Real file mappings (must match what the model was trained on) ──────────
        IMAGE_MAP = {
            "Edwin":   {"neutral": "IMG_5552.jpg",       "smiling": "IMG_5553.jpg"},
            "Glory":   {"neutral": "IMG_3926.JPG"},
            "Kevin":   {"neutral": "IMG_1881_fixed.jpg"},
            "Justine": {"neutral": "IMG_0802.PNG"},
        }
        AUDIO_MAP = {
            "Edwin":   {"approve": "Yes, approve.m4a",    "confirm": "Confirm Transaction.m4a"},
            "Glory":   {"approve": "Yes Approve.m4a",     "confirm": "Confirm Transaction.m4a"},
            "Kevin":   {"approve": "yes, approve.m4a",    "confirm": "yes, approve (2).m4a"},
            "Justine": {"approve": "Yes, I approve.m4a",  "confirm": "Confirm Transaction.m4a"},
        }

        def load_real_image(member, expr):
            fname = IMAGE_MAP[member][expr]
            fpath = os.path.join(IMG_DIR, member, fname)
            raw   = cv2.imread(fpath)
            if raw is None:
                raise FileNotFoundError(f"Cannot read image: {fpath}")
            return cv2.cvtColor(cv2.resize(raw, (224, 224)), cv2.COLOR_BGR2RGB)

        def load_real_audio(member, phrase):
            fname = AUDIO_MAP[member][phrase]
            fpath = os.path.join(AUDIO_DIR, member, fname)
            sig, _ = librosa.load(fpath, sr=SR)
            return sig

        # ── Team demo: authenticate all 4 members individually ───────────────────
        if args.demo == 'team':
            SEP2 = "─" * 65
            print(f"\n{'═'*65}")
            print("  TEAM DEMO — All 4 members authenticating individually")
            print(f"{'═'*65}")
            for member in MEMBERS:
                img = load_real_image(member, 'neutral')
                sig = load_real_audio(member, 'approve')
                run_pipeline(img, sig, SR, face_bundle, voice_bundle, product_bundle,
                             label=f"{member}: neutral face + approve voice  →  expects APPROVAL")
            print(f"\n{'═'*65}")
            print("  UNAUTHORIZED ATTEMPT — Stranger tries to access the system")
            print(f"{'═'*65}")
            stranger = np.random.randint(80, 200, (224, 224, 3), dtype=np.uint8)
            sig_g = load_real_audio('Glory', 'approve')
            run_pipeline(stranger, sig_g, SR, face_bundle, voice_bundle, product_bundle,
                         label="Stranger face (noise) + Glory's voice  →  expects DENIAL")
            return

        for scenario in scenarios:
            if scenario == 'authorized':
                # Edwin's real neutral photo + real "Yes, approve" recording
                img = load_real_image('Edwin', 'neutral')
                sig = load_real_audio('Edwin', 'approve')
                label = "Edwin: real face + real voice  →  expects APPROVAL"

            elif scenario == 'unauthorized':
                # Pure noise image (unknown stranger) + Glory's real voice
                img = np.random.randint(80, 200, (224, 224, 3), dtype=np.uint8)
                sig = load_real_audio('Glory', 'approve')
                label = "Stranger face (noise) + Glory's voice  →  expects DENIAL at face step"

            elif scenario == 'mismatch':
                # Edwin's real smiling photo + Justine's real voice
                img = load_real_image('Edwin', 'smiling')
                sig = load_real_audio('Justine', 'confirm')
                label = "Edwin's real face + Justine's real voice  →  expects DENIAL at voice step"

            run_pipeline(img, sig, SR, face_bundle, voice_bundle, product_bundle, label)

    else:
        # Load from files
        if not os.path.exists(args.face):
            print(f"[ERROR] Face image not found: {args.face}")
            sys.exit(1)
        if not os.path.exists(args.voice):
            print(f"[ERROR] Voice file not found: {args.voice}")
            sys.exit(1)

        raw_img = cv2.imread(args.face)
        if raw_img is None:
            print(f"[ERROR] Could not read image: {args.face}")
            sys.exit(1)
        img_rgb = cv2.cvtColor(cv2.resize(raw_img, (224, 224)), cv2.COLOR_BGR2RGB)

        signal, sr = librosa.load(args.voice, sr=SR)

        run_pipeline(img_rgb, signal, sr, face_bundle, voice_bundle, product_bundle,
                     label=f"Face: {os.path.basename(args.face)} | Voice: {os.path.basename(args.voice)}")


if __name__ == "__main__":
    main()
