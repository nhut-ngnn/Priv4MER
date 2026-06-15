import os
import sys
import torch
import pickle
import argparse
import random
import csv
import json
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.utils import set_seed, train_and_evaluate
from src.model_factory import (
    MODEL_CHOICES,
    build_model,
    get_model_display_name,
    normalize_model_name,
)
from torch.utils.data import TensorDataset, DataLoader


def _ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)


def build_tensor_dataset(data, label_map=None):
    if label_map is None:
        label_map = {}

    next_index = len(label_map)
    text_tensors, audio_tensors = [], []
    label_tensors, confidence_tensors, pseudo_flags = [], [], []

    for item in data:
        base_text = _ensure_tensor(item['text_embed'])
        base_audio = _ensure_tensor(item['audio_embed'])

        text_tensors.append(base_text)
        audio_tensors.append(base_audio)

        label_value = item.get('label', -1)
        if isinstance(label_value, str):
            normalized_label = label_value.strip().lower()
            if normalized_label not in label_map:
                label_map[normalized_label] = next_index
                next_index += 1
            label_idx = label_map[normalized_label]
        else:
            label_idx = int(label_value)

        label_tensors.append(torch.tensor(label_idx, dtype=torch.long))

        confidence_value = item.get('confidence', 1.0)
        if confidence_value is None:
            confidence_value = 1.0
        confidence_tensors.append(torch.tensor(float(confidence_value), dtype=torch.float32))
        pseudo_flags.append(torch.tensor(1.0 if item.get('is_pseudo', False) else 0.0, dtype=torch.float32))

    text_tensor = torch.stack(text_tensors)
    audio_tensor = torch.stack(audio_tensors)
    label_tensor = torch.stack(label_tensors)
    confidence_tensor = torch.stack(confidence_tensors)
    pseudo_tensor = torch.stack(pseudo_flags)

    dataset = TensorDataset(
        text_tensor,
        audio_tensor,
        label_tensor,
        confidence_tensor,
        pseudo_tensor
    )

    return dataset, label_map


def build_unlabeled_dataset(data):
    if not data:
        return None

    text_tensors = [_ensure_tensor(item['text_embed']) for item in data]
    audio_tensors = [_ensure_tensor(item['audio_embed']) for item in data]

    return TensorDataset(torch.stack(text_tensors), torch.stack(audio_tensors))


def clone_samples(samples, is_pseudo=None, confidence=None):
    cloned = []
    for item in samples:
        new_item = dict(item)
        if is_pseudo is not None:
            new_item["is_pseudo"] = is_pseudo
        if confidence is not None or new_item.get("confidence") is None:
            new_item["confidence"] = confidence if confidence is not None else 1.0
        cloned.append(new_item)
    return cloned


def split_labeled_unlabeled(samples, labeled_ratio, seed=None):
    if not samples:
        return [], []

    ratio = max(0.0, min(1.0, float(labeled_ratio)))
    if ratio <= 0.0:
        return [], clone_samples(samples, is_pseudo=True, confidence=0.0)
    if ratio >= 1.0:
        return clone_samples(samples, is_pseudo=False, confidence=1.0), []

    indices = list(range(len(samples)))
    rng = random.Random(seed) if seed is not None else random.Random()
    rng.shuffle(indices)

    labeled_count = int(ratio * len(samples))
    labeled_indices = indices[:labeled_count]
    unlabeled_indices = indices[labeled_count:]

    labeled = [
        dict(samples[idx], is_pseudo=False, confidence=1.0)
        for idx in labeled_indices
    ]
    unlabeled = [
        dict(samples[idx], is_pseudo=True, confidence=0.0)
        for idx in unlabeled_indices
    ]

    return labeled, unlabeled

def combined_loss(outputs, labels, ce_loss, confidences=None):
    logits = outputs["logits"]

    losses = ce_loss(logits, labels)   # [N]  

    if confidences is not None:
        weights = confidences.float()
        return (losses * weights).sum() / (weights.sum() + 1e-8)
    else:
        return losses


def append_csv(path, fieldnames, row):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_filenames(data_dir, dataset, num_classes):
    if dataset == "IEMOCAP":
        assert num_classes == 4, "IEMOCAP now only supports 4 classes."
        prefix = "IEMOCAP_BERT_Wav2Vec2"
    elif dataset == "MSP-IMPROV":
        assert num_classes == 4, "MSP-IMPROV uses 4 classes."
        prefix = "MSPIMPROV_BERT_Wav2Vec2"
    elif dataset == "ESD":
        assert num_classes == 5, "ESD uses 5 classes."
        prefix = "ESD_BERT_Wav2Vec2"
    elif dataset == "MELD":
        assert num_classes == 7, "MELD uses 7 classes."
        prefix = "MELD_BERT_Wav2Vec2"
    else:
        raise ValueError("Dataset must be one of 'IEMOCAP', 'MSP-IMPROV', 'ESD', or 'MELD'.")

    return {
        "train": os.path.join(data_dir, f"{prefix}_train.pkl"),
        "val": os.path.join(data_dir, f"{prefix}_val.pkl"),
        "test": os.path.join(data_dir, f"{prefix}_test.pkl"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal SER models on IEMOCAP/MSP-IMPROV/ESD/MELD with 5 seeds")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["IEMOCAP", "MSP-IMPROV", "ESD", "MELD"])
    parser.add_argument("--num_classes", type=int, required=True, choices=[4, 5, 7],
                        help="4 for IEMOCAP/MSP-IMPROV, 5 for ESD, 7 for MELD")
    parser.add_argument(
        "--model_name",
        type=str,
        default="fedalmer",
        help=f"Model name. Supported: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--modality", type=str, default="both", choices=["both", "text", "audio"],
                        help="Use both modalities or a single one by masking the other branch.")
    parser.add_argument("--supervised_epochs", type=int, default=None,
                        help="Number of epochs for the supervised pretraining stage. Defaults to --epochs.")
    parser.add_argument("--semi_epochs", type=int, default=None,
                        help="Number of epochs for the semi-supervised fine-tuning stage. Defaults to --epochs.")
    parser.add_argument("--pseudo_threshold", type=float, default=0.9,
                        help="Confidence threshold to accept pseudo-labeled samples.")
    parser.add_argument("--labeled_ratio", type=float, default=None,
                        help="Fraction of training data to treat as labeled (0-1).")
    parser.add_argument("--unlabeled_ratio", type=float, default=None,
                        help="Deprecated. Fraction of training data to treat as unlabeled.")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Seed used when splitting labeled/unlabeled data. Defaults to the current training seed.")
    parser.add_argument("--results_suffix", type=str, default=None,
                        help="Optional custom suffix for the aggregated results CSV filename.")
    parser.add_argument("--lambda_u", type=float, default=None,
                        help="Weight applied to the pseudo-labeled loss on unlabeled data.")
    parser.add_argument("--pseudo_weight", type=float, default=None,
                        help="Deprecated. Use --lambda_u instead.")
    parser.add_argument("--weak_word_dropout", type=float, default=0.1,
                        help="Word dropout probability for weak augmentation.")
    parser.add_argument("--strong_word_dropout", type=float, default=0.3,
                        help="Word dropout probability for strong augmentation.")
    parser.add_argument("--weak_audio_noise_std", type=float, default=0.1,
                        help="Gaussian noise std for audio during weak augmentation.")
    parser.add_argument("--strong_audio_noise_std", type=float, default=0.3,
                        help="Gaussian noise std for audio during strong augmentation.")
    parser.add_argument("--unlabeled_batch_size", type=int, default=None,
                        help="Batch size for the unlabeled data loader. Defaults to --batch_size.")
    parser.add_argument("--logs_root", type=str, default="logs/centralized")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Optional experiment name for centralized logs.")
    parser.add_argument("--reset_logs", action="store_true",
                        help="Clear existing centralized log CSVs before logging.")


    return parser.parse_args()


def main():
    args = parse_args()
    model_name = normalize_model_name(args.model_name)
    model_display_name = get_model_display_name(model_name)
    seeds = [42, 52, 103]
    # seeds = [42]

    supervised_epochs = args.supervised_epochs or args.epochs
    semi_epochs = args.semi_epochs or args.epochs
    labeled_ratio_value = args.labeled_ratio
    if labeled_ratio_value is None:
        if args.unlabeled_ratio is not None:
            labeled_ratio_value = 1.0 - float(args.unlabeled_ratio)
        else:
            labeled_ratio_value = 1.0
    labeled_ratio_value = max(0.0, min(1.0, float(labeled_ratio_value)))
    auto_split_enabled = labeled_ratio_value < 1.0
    use_pseudo = auto_split_enabled
    ratio_name_component = f"labeled{labeled_ratio_value:.4f}".replace(".", "p")
    lambda_u_value = args.lambda_u
    if lambda_u_value is None:
        if args.pseudo_weight is not None:
            lambda_u_value = args.pseudo_weight
        else:
            lambda_u_value = 1.0
    result_suffix = (
        f"{model_display_name}_5seeds"
        if not use_pseudo
        else f"{model_display_name}_SemiPseudo_5seeds"
    )
    result_file_name = args.results_suffix or f"{args.dataset}_{args.num_classes}class_{ratio_name_component}_{result_suffix}"
    exp_name = args.exp_name or result_file_name
    log_dir = os.path.join(args.logs_root, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    seed_log_path = os.path.join(log_dir, "seed_metrics.csv")
    summary_log_path = os.path.join(log_dir, "summary_metrics.csv")
    if args.reset_logs:
        for path in (seed_log_path, summary_log_path):
            if os.path.exists(path):
                os.remove(path)

    filenames = get_filenames(args.data_dir, args.dataset, args.num_classes)
    train_path = filenames["train"]
    with open(train_path, "rb") as f:
        train_raw_full = pickle.load(f)
    with open(filenames["val"], "rb") as f:
        val_raw_full = pickle.load(f)
    with open(filenames["test"], "rb") as f:
        test_raw_full = pickle.load(f)

    wa_list, ua_list, wf1_list, uf1_list = [], [], [], []

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Running with seed {seed} ===")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        split_seed_value = args.split_seed if (auto_split_enabled and args.split_seed is not None) else None

        if auto_split_enabled:
            if split_seed_value is None:
                split_seed_value = seed
            labeled_samples, pseudo_candidates = split_labeled_unlabeled(
                train_raw_full, labeled_ratio_value, split_seed_value
            )
        else:
            labeled_samples = clone_samples(train_raw_full, is_pseudo=False, confidence=1.0)
            pseudo_candidates = []

        label_map = {}
        train_dataset, label_map = build_tensor_dataset(labeled_samples, label_map=label_map)
        val_dataset, label_map = build_tensor_dataset(
            clone_samples(val_raw_full, is_pseudo=False, confidence=1.0),
            label_map=label_map
        )
        test_dataset, label_map = build_tensor_dataset(
            clone_samples(test_raw_full, is_pseudo=False, confidence=1.0),
            label_map=label_map
        )

        pseudo_raw = list(pseudo_candidates) if pseudo_candidates else []
        pseudo_available = len(pseudo_raw)

        model = build_model(model_name=model_name, num_classes=args.num_classes).to(device)

        labeled_count = train_dataset.tensors[0].size(0)
        val_count = val_dataset.tensors[0].size(0)
        test_count = test_dataset.tensors[0].size(0)

        train_labels = train_dataset.tensors[2]
        class_sample_count = torch.bincount(train_labels, minlength=args.num_classes).float()
        class_sample_count[class_sample_count == 0] = 1.0
        class_weights = 1.0 / class_sample_count
        class_weights = class_weights / class_weights.sum() * args.num_classes
        class_weights = class_weights.to(device)

        ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        loss_fn = lambda out, y, conf: combined_loss(out, y, ce_loss_fn, confidences=conf)

        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

        save_path_supervised = (
            f"saved_model/{args.dataset}_{args.num_classes}class_{ratio_name_component}_{model_display_name}_seed{seed}.pt"
        )
        os.makedirs(os.path.dirname(save_path_supervised), exist_ok=True)
        unlabeled_batch_size_value = args.unlabeled_batch_size or args.batch_size

        tags_base = [args.dataset, f"{args.num_classes}class", model_name]
        common_context = {
            "dataset": args.dataset,
            "num_classes": args.num_classes,
            "model_name": model_name,
            "model_display_name": model_display_name,
            "modality": args.modality,
            "batch_size": args.batch_size,
            "group": f"{args.dataset}_{args.num_classes}class_{model_name}",
            "project": f"{model_display_name}-EmotionRecognition-{args.dataset}-ablation",
            "experiment_name": f"{args.dataset}_{args.num_classes}class_{model_display_name}",
            "seed": seed,
            "pseudo_enabled": use_pseudo,
            "use_augmentation": bool(use_pseudo),
            "pseudo_threshold": args.pseudo_threshold if use_pseudo else None,
            "lambda_u": lambda_u_value if use_pseudo else None,
            "pseudo_weight": lambda_u_value if use_pseudo else None,
            "weak_word_dropout": args.weak_word_dropout if use_pseudo else None,
            "strong_word_dropout": args.strong_word_dropout if use_pseudo else None,
            "weak_audio_noise_std": args.weak_audio_noise_std if use_pseudo else None,
            "strong_audio_noise_std": args.strong_audio_noise_std if use_pseudo else None,
            "unlabeled_batch_size": unlabeled_batch_size_value if use_pseudo else None,
            "labeled_samples": labeled_count,
            "val_samples": val_count,
            "test_samples": test_count,
            "pseudo_available": pseudo_available,
            "labeled_ratio": labeled_ratio_value if auto_split_enabled else None,
            "split_seed": split_seed_value,
            "results_suffix": args.results_suffix
        }

        supervised_metrics = train_and_evaluate(
            model, train_dataset, val_dataset, test_dataset,
            optimizer, scheduler,
            loss_fn,
            epochs=supervised_epochs,
            save_path=save_path_supervised,
            seed=seed,
            batch_size=args.batch_size,
            modality=args.modality,
            log_context={
                **common_context,
                "stage": "supervised",
                "tags": tags_base + ["supervised"],
                "labeled_ratio": None,
                "use_augmentation": False
            }
        )

        final_metrics = supervised_metrics

        if use_pseudo and pseudo_available > 0:
            unlabeled_dataset = build_unlabeled_dataset(pseudo_raw)
            if unlabeled_dataset is None or len(unlabeled_dataset) == 0:
                print("No unlabeled samples available for semi-supervised stage; retaining supervised-only model for this seed.")
            else:
                optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

                ce_loss_fn_semi = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                loss_fn_semi = lambda out, y, conf: combined_loss(
                    out, y, ce_loss_fn_semi, confidences=conf
                )

                save_path_semi = (
                    f"saved_model/{args.dataset}_{args.num_classes}class_{ratio_name_component}_{model_display_name}_seed{seed}_semi.pt"
                )
                os.makedirs(os.path.dirname(save_path_semi), exist_ok=True)
                unlabeled_cfg = {
                    "pseudo_weight": lambda_u_value,
                    "lambda_u": lambda_u_value,
                    "pseudo_threshold": args.pseudo_threshold,
                    "weak_word_dropout": args.weak_word_dropout,
                    "strong_word_dropout": args.strong_word_dropout,
                    "weak_audio_noise_std": args.weak_audio_noise_std,
                    "strong_audio_noise_std": args.strong_audio_noise_std,
                    "unlabeled_batch_size": unlabeled_batch_size_value
                }

                final_metrics = train_and_evaluate(
                    model, train_dataset, val_dataset, test_dataset,
                    optimizer, scheduler,
                    loss_fn_semi,
                    epochs=semi_epochs,
                    save_path=save_path_semi,
                    seed=seed,
                    batch_size=args.batch_size,
                    modality=args.modality,
                    log_context={
                        **common_context,
                        "stage": "semi_supervised",
                        "tags": tags_base + ["semi", "pseudo"],
                        "use_augmentation": True,
                        "combined_train_samples": labeled_count,
                        "lambda_u": lambda_u_value,
                        "pseudo_weight": lambda_u_value
                    },
                    unlabeled_dataset=unlabeled_dataset,
                    unlabeled_cfg=unlabeled_cfg
                )
        elif use_pseudo:
            print("No unlabeled samples available for pseudo labeling; skipping semi-supervised stage for this seed.")

        print(
            f"Seed {seed} - Final WA: {final_metrics['test_WA']:.4f}, UA: {final_metrics['test_UA']:.4f}, "
            f"WF1: {final_metrics['test_WF1']:.4f}, UF1: {final_metrics['test_UF1']:.4f}"
        )

        wa_list.append(final_metrics["test_WA"])
        ua_list.append(final_metrics["test_UA"])
        wf1_list.append(final_metrics["test_WF1"])
        uf1_list.append(final_metrics["test_UF1"])

        append_csv(seed_log_path, [
            "seed",
            "stage",
            "labeled_ratio",
            "test_WA",
            "test_UA",
            "test_WF1",
            "test_UF1",
        ], {
            "seed": seed,
            "stage": "semi_supervised" if use_pseudo else "supervised",
            "labeled_ratio": labeled_ratio_value if auto_split_enabled else 1.0,
            "test_WA": final_metrics["test_WA"],
            "test_UA": final_metrics["test_UA"],
            "test_WF1": final_metrics["test_WF1"],
            "test_UF1": final_metrics["test_UF1"],
        })

    print("\n=== Average Results over 5 seeds ===")
    print(f"Avg WA:  {np.mean(wa_list):.4f}, {np.std(wa_list, ddof=1):.4f}")
    print(f"Avg UA:  {np.mean(ua_list):.4f}, {np.std(ua_list, ddof=1):.4f}")
    print(f"Avg WF1: {np.mean(wf1_list):.4f}, {np.std(wf1_list, ddof=1):.4f}")
    print(f"Avg UF1: {np.mean(uf1_list):.4f}, {np.std(uf1_list, ddof=1):.4f}")

    results_df = pd.DataFrame({
        "Metric": ["WA", "UA", "WF1", "UF1"],
        "Mean": [np.mean(wa_list), np.mean(ua_list), np.mean(wf1_list), np.mean(uf1_list)],
        "Std": [np.std(wa_list, ddof=1), np.std(ua_list, ddof=1), np.std(wf1_list, ddof=1), np.std(uf1_list, ddof=1)]
    })

    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", f"{result_file_name}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    results_df.to_csv(summary_log_path, index=False)
    print(f"Centralized logs saved to {log_dir}")

    summary_payload = {
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "model_name": model_name,
        "model_display_name": model_display_name,
        "labeled_ratio": labeled_ratio_value,
        "seed_list": seeds,
        "num_seeds": len(seeds),
        "metrics": {},
        "results_csv": os.path.abspath(results_path),
        "seed_metrics_csv": os.path.abspath(seed_log_path),
        "summary_metrics_csv": os.path.abspath(summary_log_path),
        "logs_dir": os.path.abspath(log_dir),
    }
    for _, row in results_df.iterrows():
        metric_name = str(row["Metric"])
        summary_payload["metrics"][metric_name] = {
            "mean": float(row["Mean"]),
            "std": float(row["Std"]),
        }

    summary_json_path = os.path.join(log_dir, "summary_metrics.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

if __name__ == "__main__":
    main()
