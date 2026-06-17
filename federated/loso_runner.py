import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import shutil
from pathlib import Path

import torch
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model_factory import MODEL_CHOICES, normalize_model_name


DEFAULT_SEEDS = [42, 52, 103, 128, 923]


def _run(cmd, cwd=None):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _parse_list(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_seeds(value):
    seeds = _parse_list(value)
    parsed = []
    for item in seeds:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return parsed


def _parse_int_list(value):
    items = _parse_list(value)
    parsed = []
    for item in items:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return parsed


def _normalize_eval_checkpoint(value):
    if not value:
        return "best"
    value = str(value).strip().lower()
    if value == "lastest":
        value = "latest"
    if value not in ("best", "latest"):
        return "latest"
    return value


def _collect_msp_sessions(data_root):
    audio_root = os.path.join(data_root, "Audio")
    if not os.path.isdir(audio_root):
        raise FileNotFoundError(f"MSP-IMPROV audio root not found: {audio_root}")
    pattern = re.compile(r"MSP-IMPROV-S(\d{2})[A-Za-z]-[FM]\d{2}-")
    path_pattern = re.compile(r"[\\/](?:audio[\\/])?session(\d+)[\\/]", flags=re.IGNORECASE)
    sessions = set()
    for wav_path in Path(audio_root).rglob("*"):
        if not wav_path.is_file() or wav_path.suffix.lower() != ".wav":
            continue
        path_match = path_pattern.search(str(wav_path))
        if path_match:
            sessions.add(int(path_match.group(1)))
            continue
        name_match = pattern.match(wav_path.stem)
        if name_match:
            sessions.add(int(name_match.group(1)))
    return sorted(sessions)


def _load_yaml_config(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = {}
    defaults["dataset"] = cfg.get("dataset")
    defaults["num_classes"] = cfg.get("num_classes")
    model_cfg = cfg.get("model", {}) or {}
    defaults["model_name"] = model_cfg.get("name") or cfg.get("model_name")

    paths = cfg.get("paths", {}) or {}
    defaults["data_root"] = paths.get("data_root")
    defaults["out_base"] = paths.get("out_base") or paths.get("clients_root")
    defaults["features_base"] = paths.get("features_base") or paths.get("features_root")
    defaults["checkpoints_root"] = paths.get("checkpoints_root")
    defaults["logs_root"] = paths.get("logs_root")

    loso = cfg.get("loso", {}) or {}
    defaults["val_session"] = loso.get("val_session")
    defaults["test_sessions"] = loso.get("test_sessions")
    defaults["split_by"] = loso.get("split_by")
    defaults["num_clients"] = loso.get("num_clients")
    defaults["labeled_ratio"] = loso.get("labeled_ratio")
    defaults["client_val_ratio"] = loso.get("client_val_ratio")
    defaults["min_labeled_per_client"] = loso.get("min_labeled_per_client")
    defaults["seed"] = loso.get("seed")
    defaults["exp_prefix"] = loso.get("exp_prefix")
    defaults["eval_checkpoint"] = loso.get("eval_checkpoint")
    defaults["skip_eval"] = loso.get("skip_eval")
    defaults["run_id"] = loso.get("run_id")
    defaults["clean_metadata"] = loso.get("clean_metadata")
    defaults["clean_features"] = loso.get("clean_features")
    defaults["seeds"] = loso.get("seeds")

    fed = cfg.get("federated", {}) or {}
    defaults["rounds_pretrain"] = fed.get("rounds_pretrain")
    defaults["rounds_ssl"] = fed.get("rounds_ssl")
    defaults["local_epochs_pretrain"] = fed.get("local_epochs_pretrain")
    defaults["local_epochs_ssl"] = fed.get("local_epochs_ssl")
    defaults["batch_size"] = fed.get("batch_size")
    defaults["modality"] = fed.get("modality")
    defaults["lr"] = fed.get("lr")
    defaults["weight_decay"] = fed.get("weight_decay")
    defaults["weight_by"] = fed.get("weight_by")
    defaults["best_metric"] = fed.get("best_metric")
    defaults["fl_method"] = fed.get("fl_method")
    defaults["model_name"] = fed.get("model_name") or defaults["model_name"]
    defaults["fedprox_mu"] = fed.get("fedprox_mu")
    defaults["num_workers"] = fed.get("num_workers")
    defaults["skip_eval"] = fed.get("skip_eval", defaults.get("skip_eval"))

    ssl = cfg.get("ssl", {}) or {}
    defaults["tau"] = ssl.get("tau")
    defaults["lambda_u"] = ssl.get("lambda_u")
    defaults["weak_word_dropout"] = ssl.get("weak_word_dropout")
    defaults["strong_word_dropout"] = ssl.get("strong_word_dropout")
    defaults["weak_audio_noise_std"] = ssl.get("weak_audio_noise_std")
    defaults["strong_audio_noise_std"] = ssl.get("strong_audio_noise_std")
    defaults["unlabeled_batch_size"] = ssl.get("unlabeled_batch_size")
    extract = cfg.get("extract", {}) or {}
    defaults["nrc_lexicon"] = extract.get("nrc_lexicon")

    return defaults


def _list_clients(clients_root):
    return sorted(
        p.name for p in Path(clients_root).glob("client_*")
        if p.is_dir()
    )


def _read_aggregate_metrics(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[-1]


def _aggregate_results(rows):
    keys = ["global_WA", "global_UA", "global_WF1", "global_UF1"]
    total_samples = sum(int(row["test_samples"]) for row in rows)
    if total_samples <= 0:
        macro = {key: 0.0 for key in keys}
        weighted = {key: 0.0 for key in keys}
        return macro, weighted, total_samples
    macro = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in keys
    }
    weighted = {
        key: sum(float(row[key]) * int(row["test_samples"]) for row in rows) / total_samples
        for key in keys
    }
    return macro, weighted, total_samples


def _mean_std(values):
    count = len(values)
    if count == 0:
        return 0.0, 0.0
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    var = sum((val - mean) ** 2 for val in values) / (count - 1)
    return mean, math.sqrt(var)


def _weighted_mean_std(values, weights):
    if not values:
        return 0.0, 0.0
    total = float(sum(weights))
    if total <= 0:
        return 0.0, 0.0
    mean = sum(val * weight for val, weight in zip(values, weights)) / total
    if len(values) < 2:
        return mean, 0.0
    var = sum(weight * (val - mean) ** 2 for val, weight in zip(values, weights)) / total
    return mean, math.sqrt(var)


def _aggregate_fold_results(rows):
    keys = ["global_WA", "global_UA", "global_WF1", "global_UF1"]
    if not rows:
        empty = {key: 0.0 for key in keys}
        return empty, empty, empty, empty, 0

    totals = [int(row.get("test_samples", 0)) for row in rows]
    total_samples = sum(totals)

    macro_avg = {}
    macro_std = {}
    weighted_avg = {}
    weighted_std = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        mean, std = _mean_std(values)
        macro_avg[key] = mean
        macro_std[key] = std

        w_mean, w_std = _weighted_mean_std(values, totals)
        weighted_avg[key] = w_mean
        weighted_std[key] = w_std

    return macro_avg, macro_std, weighted_avg, weighted_std, total_samples


def _resolve_best_checkpoint(ckpt_root):
    if not os.path.isdir(ckpt_root):
        return None
    meta_path = os.path.join(ckpt_root, "best_meta.pt")
    if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location="cpu")
        best_path = meta.get("best_path")
        if best_path and os.path.isfile(best_path):
            return best_path
        best_round = meta.get("best_round")
        if best_round is not None:
            candidate = os.path.join(ckpt_root, f"global_round_best_{best_round}.pt")
            if os.path.isfile(candidate):
                return candidate

    best_candidates = []
    for name in os.listdir(ckpt_root):
        if not (name.startswith("global_round_best_") and name.endswith(".pt")):
            continue
        parts = name.split("_")
        if len(parts) < 4:
            continue
        round_part = parts[-1].replace(".pt", "")
        try:
            round_idx = int(round_part)
        except ValueError:
            continue
        best_candidates.append((round_idx, os.path.join(ckpt_root, name)))
    if not best_candidates:
        return None
    best_candidates.sort(key=lambda x: x[0], reverse=True)
    return best_candidates[0][1]


def _ensure_eval_metrics(args, exp_name, stage, clients_root, features_root):
    eval_checkpoint = _normalize_eval_checkpoint(args.eval_checkpoint)
    eval_dir = os.path.join(args.logs_root, exp_name, "eval", stage, eval_checkpoint)
    metrics_path = os.path.join(eval_dir, "aggregate_metrics.csv")
    if os.path.isfile(metrics_path):
        return metrics_path

    ckpt_root = os.path.join(args.checkpoints_root, exp_name, stage)
    if eval_checkpoint == "latest":
        ckpt_path = os.path.join(ckpt_root, "global_round_latest.pt")
    else:
        ckpt_path = _resolve_best_checkpoint(ckpt_root)

    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Missing checkpoint for {eval_checkpoint}: {ckpt_path or ckpt_root}"
        )

    from federated.evaluate import evaluate_checkpoint

    evaluate_checkpoint(
        dataset=args.dataset,
        num_classes=args.num_classes,
        model_name=args.model_name,
        modality=args.modality,
        clients_root=clients_root,
        features_root=features_root,
        checkpoint=ckpt_path,
        batch_size=args.batch_size,
        num_clients=args.num_clients if args.num_clients and args.num_clients > 0 else None,
        num_workers=args.num_workers,
        output_dir=eval_dir,
        reset_logs=True,
    )

    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Missing eval metrics after evaluation: {metrics_path}")

    return metrics_path


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None)
    known_args, _ = base_parser.parse_known_args()
    cfg_defaults = _load_yaml_config(known_args.config)

    parser = argparse.ArgumentParser(description="Run session-based LOSO.")
    parser.add_argument("--config", type=str, default=known_args.config)
    parser.add_argument("--dataset", type=str, default="IEMOCAP", choices=["IEMOCAP", "MSP-IMPROV"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split_by", type=str, default="session", choices=["session"])
    parser.add_argument("--val_session", type=int, default=None)
    parser.add_argument("--test_sessions", type=str, default=None)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--labeled_ratio", type=float, default=1.0)
    parser.add_argument("--client_val_ratio", type=float, default=0.1)
    parser.add_argument("--min_labeled_per_client", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds for training runs.")
    parser.add_argument("--out_base", type=str, default="metadata/IEMOCAP_loso/clients")
    parser.add_argument("--features_base", type=str, default="features/IEMOCAP/clients")
    parser.add_argument("--exp_prefix", type=str, default="IEMOCAP_loso")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Optional tag to isolate metadata/features/logs per run.")
    parser.add_argument("--clean_metadata", action="store_true",
                        help="Delete existing metadata folder for each fold before preprocess.")
    parser.add_argument("--clean_features", action="store_true",
                        help="Delete existing feature cache folder for each fold before extraction.")

    parser.add_argument("--num_classes", type=int, default=4, choices=[4, 5, 7])
    parser.add_argument(
        "--model_name",
        type=str,
        default="fedalmer",
        help=f"Model name. Supported: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument("--rounds_pretrain", type=int, default=1)
    parser.add_argument("--rounds_ssl", type=int, default=1)
    parser.add_argument("--local_epochs_pretrain", type=int, default=1)
    parser.add_argument("--local_epochs_ssl", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--modality", type=str, default="both", choices=["both", "text", "audio"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--weak_word_dropout", type=float, default=0.1)
    parser.add_argument("--strong_word_dropout", type=float, default=0.3)
    parser.add_argument("--weak_audio_noise_std", type=float, default=0.1)
    parser.add_argument("--strong_audio_noise_std", type=float, default=0.3)
    parser.add_argument("--unlabeled_batch_size", type=int, default=None)
    parser.add_argument("--weight_by", type=str, default="total", choices=["total", "labeled"])
    parser.add_argument(
        "--best_metric",
        type=str,
        default="global_UA",
        choices=["global_WA", "global_UA", "global_WF1", "global_UF1"],
    )
    parser.add_argument("--fl_method", type=str, default="fedavg",
                        choices=["fedavg", "fedprox"],
                        help="Federated optimization method.")
    parser.add_argument("--fedprox_mu", type=float, default=0.0,
                        help="FedProx proximal term weight (mu).")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--checkpoints_root", type=str, default="checkpoints/federated")
    parser.add_argument("--logs_root", type=str, default="logs/federated")
    parser.add_argument("--eval_checkpoint", type=str, default="best", choices=["best", "latest"])
    parser.add_argument("--resume_global", type=str, default=None)
    parser.add_argument("--nrc_lexicon", type=str, default=None,
                        help="Optional NRC Emotion Lexicon word-level TSV path for feature extraction.")

    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip checkpoint evaluation and LOSO metric aggregation.")

    for key, value in cfg_defaults.items():
        if value is not None:
            parser.set_defaults(**{key: value})
    args = parser.parse_args()
    args.model_name = normalize_model_name(args.model_name)
    if args.split_by != "session":
        parser.error("Speaker-based LOSO has been removed. Use split_by: session.")
    needs_data_root = (not args.skip_preprocess) or (not args.skip_features) or (args.dataset == "MSP-IMPROV")
    if needs_data_root and not args.data_root:
        parser.error(
            "Missing required arg: data_root "
            "(needed when preprocessing/features are enabled, or dataset=MSP-IMPROV)."
        )
    return args


def main():
    args = parse_args()
    args.eval_checkpoint = _normalize_eval_checkpoint(args.eval_checkpoint)
    seed_list = _parse_seeds(args.seeds) or DEFAULT_SEEDS

    if args.dataset == "IEMOCAP":
        session_ids = [1, 2, 3, 4, 5]
    elif args.dataset == "MSP-IMPROV":
        session_ids = _collect_msp_sessions(args.data_root)
        if len(session_ids) < 3:
            raise ValueError("MSP-IMPROV requires at least 3 sessions for LOSO.")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    test_sessions = _parse_int_list(args.test_sessions)
    if not test_sessions:
        test_sessions = list(session_ids)
    missing = [sess for sess in test_sessions if sess not in session_ids]
    if missing:
        raise ValueError(f"Unknown test sessions: {missing}")
    num_clients = args.num_clients if args.num_clients and args.num_clients > 0 else None
    stage = "ssl" if args.rounds_ssl > 0 else "pretrain"
    fold_results = []
    seed_results = []

    run_tag = str(args.run_id).strip() if args.run_id else ""

    fold_items = [{"test_session": sess} for sess in test_sessions]

    for fold in fold_items:
        test_spk = None
        val_spk = None
        test_session = fold.get("test_session")
        val_session = None
        val_pct = int(round(float(args.client_val_ratio) * 100))
        fold_name = f"test_session{test_session}_clientval{val_pct}"

        fold_seed_rows = []

        if run_tag:
            clients_root = os.path.join(args.out_base, run_tag, fold_name)
            features_root = os.path.join(args.features_base, run_tag, fold_name)
        else:
            clients_root = os.path.join(args.out_base, fold_name)
            features_root = os.path.join(args.features_base, fold_name)

        if args.clean_metadata and os.path.isdir(clients_root):
            shutil.rmtree(clients_root, ignore_errors=True)
        if args.clean_features and os.path.isdir(features_root):
            shutil.rmtree(features_root, ignore_errors=True)

        if not args.skip_preprocess:
            cmd = [
                sys.executable, "federated/preprocess.py",
                "--dataset", args.dataset,
                "--data_root", args.data_root,
                "--out_dir", clients_root,
                "--split_by", args.split_by,
                "--labeled_ratio", str(args.labeled_ratio),
                "--client_val_ratio", str(args.client_val_ratio),
                "--min_labeled_per_client", str(args.min_labeled_per_client),
                "--seed", str(args.seed),
            ]
            cmd += [
                "--test_by", "session",
                "--val_by", "session",
                "--test_session", str(test_session),
            ]
            if num_clients is not None:
                cmd += ["--num_clients", str(num_clients)]
            _run(cmd)

        if not args.skip_features:
            clients = _list_clients(clients_root)
            if num_clients is not None:
                clients = clients[:num_clients]
            if not clients:
                raise ValueError(f"No clients found under {clients_root}")
            for client_id in clients:
                client_dir = os.path.join(clients_root, client_id)
                out_dir = os.path.join(features_root, client_id)
                cmd = [
                    sys.executable, "feature_extract/extract_feature.py",
                    "--dataset", args.dataset,
                    "--client_dir", client_dir,
                    "--out_dir", out_dir,
                    "--wav_base", args.data_root,
                ]
                if args.nrc_lexicon:
                    cmd += ["--nrc_lexicon", args.nrc_lexicon]
                _run(cmd)

        for train_seed in seed_list:
            if run_tag:
                if args.model_name == "fedalmer":
                    exp_name = f"{args.exp_prefix}_{run_tag}_{fold_name}_seed{train_seed}"
                else:
                    exp_name = f"{args.exp_prefix}_{args.model_name}_{run_tag}_{fold_name}_seed{train_seed}"
            else:
                if args.model_name == "fedalmer":
                    exp_name = f"{args.exp_prefix}_{fold_name}_seed{train_seed}"
                else:
                    exp_name = f"{args.exp_prefix}_{args.model_name}_{fold_name}_seed{train_seed}"

            if not args.skip_train:
                cmd = [sys.executable, "-m", "federated.run_federated"]
                if args.config:
                    cmd += ["--config", args.config]
                cmd += [
                    "--dataset", args.dataset,
                    "--num_classes", str(args.num_classes),
                    "--model_name", args.model_name,
                    "--clients_root", clients_root,
                    "--features_root", features_root,
                    "--rounds_pretrain", str(args.rounds_pretrain),
                    "--rounds_ssl", str(args.rounds_ssl),
                    "--local_epochs_pretrain", str(args.local_epochs_pretrain),
                    "--local_epochs_ssl", str(args.local_epochs_ssl),
                    "--batch_size", str(args.batch_size),
                    "--modality", args.modality,
                    "--lr", str(args.lr),
                    "--weight_decay", str(args.weight_decay),
                    "--tau", str(args.tau),
                    "--lambda_u", str(args.lambda_u),
                    "--weak_word_dropout", str(args.weak_word_dropout),
                    "--strong_word_dropout", str(args.strong_word_dropout),
                    "--weak_audio_noise_std", str(args.weak_audio_noise_std),
                    "--strong_audio_noise_std", str(args.strong_audio_noise_std),
                    "--weight_by", args.weight_by,
                    "--best_metric", args.best_metric,
                    "--fl_method", args.fl_method,
                    "--fedprox_mu", str(args.fedprox_mu),
                    "--num_workers", str(args.num_workers),
                    "--checkpoints_root", args.checkpoints_root,
                    "--logs_root", args.logs_root,
                    "--exp_name", exp_name,
                    "--seed", str(train_seed),
                ]
                if args.unlabeled_batch_size is not None:
                    cmd += ["--unlabeled_batch_size", str(args.unlabeled_batch_size)]
                if args.resume_global:
                    cmd += ["--resume_global", args.resume_global]
                if num_clients is not None:
                    cmd += ["--num_clients", str(num_clients)]
                _run(cmd)

            if args.skip_eval:
                continue

            metrics_path = _ensure_eval_metrics(
                args,
                exp_name,
                stage,
                clients_root,
                features_root,
            )

            row = _read_aggregate_metrics(metrics_path)
            row["split_by"] = args.split_by
            row["test_speaker"] = test_spk
            row["val_speaker"] = val_spk
            row["test_session"] = test_session
            row["val_session"] = val_session
            row["exp_name"] = exp_name
            row["seed"] = train_seed
            row["fold"] = fold_name
            seed_results.append(row)
            fold_seed_rows.append(row)

        metrics = ["global_WA", "global_UA", "global_WF1", "global_UF1"]
        fold_summary = {
            "fold": fold_name,
            "split_by": args.split_by,
            "test_speaker": test_spk,
            "val_speaker": val_spk,
            "test_session": test_session,
            "val_session": val_session,
            "num_seeds": len(fold_seed_rows),
            "seed_list": ",".join(str(s) for s in seed_list),
            "checkpoint": args.eval_checkpoint,
        }
        test_samples = int(fold_seed_rows[0]["test_samples"]) if fold_seed_rows else 0
        fold_summary["test_samples"] = test_samples
        for key in metrics:
            values = [float(row.get(key, 0.0)) for row in fold_seed_rows]
            mean, std = _mean_std(values)
            fold_summary[key] = mean
            fold_summary[f"{key}_std"] = std

        fold_results.append(fold_summary)

    macro, macro_std, weighted, weighted_std, total_samples = _aggregate_fold_results(fold_results)

    exp_prefix_for_logs = args.exp_prefix
    if args.model_name != "fedalmer":
        exp_prefix_for_logs = f"{args.exp_prefix}_{args.model_name}"

    loso_dir = os.path.join(args.logs_root, "loso", exp_prefix_for_logs)
    if run_tag:
        loso_dir = os.path.join(loso_dir, run_tag)
    os.makedirs(loso_dir, exist_ok=True)
    per_seed_path = os.path.join(loso_dir, "per_seed_metrics.csv")
    with open(per_seed_path, "w", newline="") as f:
        fieldnames = [
            "fold", "exp_name", "seed",
            "split_by",
            "test_speaker", "val_speaker",
            "test_session", "val_session",
            "test_samples", "global_WA", "global_UA", "global_WF1", "global_UF1",
            "checkpoint",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in seed_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    per_fold_path = os.path.join(loso_dir, "per_fold_metrics.csv")
    with open(per_fold_path, "w", newline="") as f:
        fieldnames = [
            "fold", "split_by",
            "test_speaker", "val_speaker",
            "test_session", "val_session",
            "num_seeds", "seed_list",
            "test_samples",
            "global_WA", "global_WA_std",
            "global_UA", "global_UA_std",
            "global_WF1", "global_WF1_std",
            "global_UF1", "global_UF1_std",
            "checkpoint",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    per_fold_json_path = os.path.join(loso_dir, "per_fold_metrics.json")
    with open(per_fold_json_path, "w") as f:
        json.dump(fold_results, f, indent=2)

    summary = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "split_by": args.split_by,
        "val_speaker": None,
        "val_session": None,
        "num_folds": len(fold_results),
        "num_seeds": len(seed_list),
        "seed_list": seed_list,
        "stage": stage,
        "checkpoint_type": args.eval_checkpoint,
        "skip_eval": args.skip_eval,
        "macro_avg": macro,
        "macro_std": macro_std,
        "weighted_avg": weighted,
        "weighted_std": weighted_std,
        "total_test_samples": total_samples,
    }
    summary_path = os.path.join(loso_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("LOSO summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
