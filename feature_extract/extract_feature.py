import os
import sys
import json
import time
import pickle
import warnings
import argparse

import numpy as np
import torch
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")
p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
print(p)
sys.path.append(p)

from src.feature_extract.config import (
    PKL_DIR, OUTPUT_DIR, device,
    TOKENIZER, AUDIO_PROCESSOR,
    TEXT_MODEL, AUDIO_MODEL
)



def _load_audio(audio_path):
    array, sr = sf.read(audio_path)

    if isinstance(array, np.ndarray):
        waveform = array.astype(np.float32)
    else:
        waveform = np.array(array, dtype=np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000

    return np.ascontiguousarray(waveform), sr


def _extract_audio_embedding(waveform, sr, processor, model, device, source=None):
    try:
        inputs = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pooled, _ = model(inputs["input_values"])
        return pooled.squeeze().cpu()
    except Exception as e:
        prefix = f"{source}: " if source else ""
        print(f"[ERROR] Audio failed: {prefix}({e})")
        return None


def extract_audio_features(audio_path, processor, model, device):
    try:
        waveform, sr = _load_audio(audio_path)
    except Exception as e:
        print(f"[ERROR] Audio failed: {audio_path} ({e})")
        return None
    return _extract_audio_embedding(
        waveform,
        sr,
        processor,
        model,
        device,
        source=audio_path
    )

def extract_text_features(text, tokenizer, model, device):
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        with torch.no_grad():
            pooled, _ = model(inputs["input_ids"], inputs["attention_mask"]) 
        return pooled.squeeze().cpu()
    except Exception as e:
        print(f"[ERROR] Text failed: {text[:30]}... ({e})")
        return None

def process_single_sample(audio_path, text, label, is_pseudo=False, confidence=None, skip_text=False):
    audio_embed = extract_audio_features(audio_path, AUDIO_PROCESSOR, AUDIO_MODEL, device)
    if audio_embed is None:
        return None

    if skip_text:
        text_embed = torch.zeros_like(audio_embed)
    else:
        text_embed = extract_text_features(text, TOKENIZER, TEXT_MODEL, device)
        if text_embed is None:
            return None

    return {
        "text_embed": text_embed,
        "audio_embed": audio_embed,
        "label": label,
        "is_pseudo": is_pseudo,
        "confidence": confidence,
        "sample_id": os.path.basename(audio_path),
        "raw_text": text,
        "audio_path": audio_path
    }


def process_dataset(pkl_path, wav_base, output_path, pseudo=False, skip_text=False):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    processed_samples = []
    print(f"Processing {len(data)} samples from {pkl_path}")

    for item in tqdm(data, desc=f"Processing {os.path.basename(pkl_path)}"):
        if isinstance(item, dict):
            filename = (
                item.get("filename")
                or item.get("audio_path")
                or item.get("path")
                or item.get("file_path")
                or item.get("filepath")
                or item.get("wav_path")
            )
        elif isinstance(item, (tuple, list)):
            filename = item[0]
        else:
            raise TypeError(f"Unexpected item type: {type(item)}")

        if os.path.isabs(filename):
            audio_path = filename
        else:
            audio_path = os.path.join(wav_base, os.path.basename(filename))

        if isinstance(item, dict):
            text = item.get("text") or item.get("transcript") or item.get("utterance")
        else:
            text = item[1] if len(item) > 1 else ""

        label = None
        conf = None
        if isinstance(item, dict):
            if pseudo:
                label = item.get("pseudo_label")
                if label is None:
                    label = item.get("label", item.get("emotion"))
                conf = item.get("confidence", None)
            else:
                label = item.get("label", item.get("emotion"))
                conf = item.get("confidence", None)
        elif isinstance(item, (tuple, list)) and len(item) > 2:
            label = item[2]

        sample = process_single_sample(
            audio_path,
            text,
            label,
            is_pseudo=pseudo,
            confidence=conf,
            skip_text=skip_text,
        )
        if sample is not None:
            processed_samples.append(sample)
        else:
            print(f"[SKIP] Failed to process: {audio_path}")

    with open(output_path, "wb") as f:
        pickle.dump(processed_samples, f)

    print(f"Saved processed data to: {output_path}")
    print(f"Total processed samples: {len(processed_samples)}")
    return processed_samples


def _write_manifest(out_dir, manifest):
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to: {manifest_path}")


def _process_client_dir(client_dir, out_dir, wav_base):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "client_dir": os.path.abspath(client_dir),
        "out_dir": os.path.abspath(out_dir),
        "splits": {},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    split_specs = [
        ("train_labeled", "train_labeled.pkl", "train_labeled_features.pkl", False),
        ("train_unlabeled", "train_unlabeled.pkl", "train_unlabeled_features.pkl", False),
        ("val", "val.pkl", "val_features.pkl", False),
        ("test", "test.pkl", "test_features.pkl", False),
    ]

    for split_name, input_name, output_name, pseudo_flag in split_specs:
        input_path = os.path.join(client_dir, input_name)
        if not os.path.isfile(input_path):
            print(f"[WARN] Missing {split_name} input at {input_path}; skipping.")
            continue
        output_path = os.path.join(out_dir, output_name)
        processed = process_dataset(
            input_path,
            wav_base,
            output_path,
            pseudo=pseudo_flag,
            skip_text=False,
        )
        sample_count = len(processed)
        text_dim = None
        audio_dim = None
        if sample_count > 0:
            text_dim = list(processed[0]["text_embed"].shape)
            audio_dim = list(processed[0]["audio_embed"].shape)
        manifest["splits"][split_name] = {
            "input_path": os.path.abspath(input_path),
            "output_path": os.path.abspath(output_path),
            "count": sample_count,
            "text_dim": text_dim,
            "audio_dim": audio_dim,
        }

    _write_manifest(out_dir, manifest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["IEMOCAP", "MSP-IMPROV", "ESD", "MELD"],
        help="Dataset to process",
    )
    parser.add_argument("--pseudo", action="store_true", help="Flag to process dataset as pseudo-labeled")
    parser.add_argument("--wav_base", type=str, default=None, help="Root directory containing the waveform files for the dataset.")
    parser.add_argument("--client_dir", type=str, default=None, help="Directory containing per-client metadata PKLs.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for extracted features.")
    parser.add_argument("--pkl_dir", type=str, default=None,
                        help="Directory containing train.pkl/val.pkl/test.pkl (non-client mode).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for extracted features (non-client mode).")
    args = parser.parse_args()

    if args.client_dir:
        if args.wav_base is None:
            raise ValueError("Please provide --wav_base to specify the audio root directory.")
        out_dir = args.out_dir or os.path.join(
            OUTPUT_DIR, "clients", os.path.basename(args.client_dir.rstrip("/"))
        )
        print("Starting client feature extraction...")
        print(f"Using device: {device}")
        _process_client_dir(args.client_dir, out_dir, args.wav_base)
        return

    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print("Starting feature extraction...")
    print(f"Using device: {device}")

    datasets = []
    if args.dataset == "IEMOCAP":
        pkl_prefix = "IEMOCAP"
        if args.pkl_dir:
            datasets = [
                ("train", os.path.join(args.pkl_dir, "train.pkl"), f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   os.path.join(args.pkl_dir, "val.pkl"),   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  os.path.join(args.pkl_dir, "test.pkl"),  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
        else:
            datasets = [
                ("train", f"{pkl_prefix}_preprocessed/train.pkl", f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   f"{pkl_prefix}_preprocessed/val.pkl",   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  f"{pkl_prefix}_preprocessed/test.pkl",  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
    elif args.dataset == "MSP-IMPROV":
        pkl_prefix = "MSPIMPROV"
        if args.pkl_dir:
            datasets = [
                ("train", os.path.join(args.pkl_dir, "train.pkl"), f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   os.path.join(args.pkl_dir, "val.pkl"),   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  os.path.join(args.pkl_dir, "test.pkl"),  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
        else:
            raise ValueError("Please provide --pkl_dir for MSP-IMPROV feature extraction.")
    elif args.dataset == "ESD":
        pkl_prefix = "ESD"
        if args.pkl_dir:
            datasets = [
                ("train", os.path.join(args.pkl_dir, "train.pkl"), f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   os.path.join(args.pkl_dir, "val.pkl"),   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  os.path.join(args.pkl_dir, "test.pkl"),  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
        else:
            datasets = [
                ("train", f"{pkl_prefix}_preprocessed/train.pkl", f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   f"{pkl_prefix}_preprocessed/val.pkl",   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  f"{pkl_prefix}_preprocessed/test.pkl",  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
    elif args.dataset == "MELD":
        pkl_prefix = "MELD"
        if args.pkl_dir:
            datasets = [
                ("train", os.path.join(args.pkl_dir, "train.pkl"), f"{pkl_prefix}_BERT_Wav2Vec2_train.pkl", False),
                ("val",   os.path.join(args.pkl_dir, "val.pkl"),   f"{pkl_prefix}_BERT_Wav2Vec2_val.pkl", False),
                ("test",  os.path.join(args.pkl_dir, "test.pkl"),  f"{pkl_prefix}_BERT_Wav2Vec2_test.pkl", False),
            ]
        else:
            raise ValueError("Please provide --pkl_dir for MELD feature extraction.")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    for split_name, pkl_file, output_file, skip_text in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split: {pkl_file}")
        print(f"{'='*50}")
        wav_base = args.wav_base
        if wav_base is None:
            raise ValueError("Please provide --wav_base to specify the audio root directory.")
        pkl_path = pkl_file if args.pkl_dir else os.path.join(PKL_DIR, pkl_file)
        output_path = os.path.join(output_dir, output_file)
        process_dataset(
            pkl_path,
            wav_base,
            output_path,
            pseudo=args.pseudo if split_name == "train" else False,
            skip_text=skip_text,
        )

if __name__ == "__main__":
    main()
