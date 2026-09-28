"""Portable PaddleX fine-tuning stages; never train as a side effect of importing.

python src/train.py check --config configs/finetune.yaml
python src/train.py train --config configs/finetune.yaml --physical-gpu-id 5
python src/train.py evaluate --config configs/finetune.yaml
python src/train.py export --config configs/finetune.yaml

Use --dry-run to inspect commands without imports, GPU access, or filesystem writes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


MODEL_NAME = "PP-LCNet_x1_0_textline_ori"
MODEL_CONFIG_RELATIVE = (
    "paddlex/configs/modules/textline_orientation/"
    "PP-LCNet_x1_0_textline_ori.yaml"
)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    required = {
        "model_name", "paddlex_commit", "paddleclas_commit", "paddlex_dir",
        "paddleclas_dir", "pretrained_weights", "dataset_dir", "output_dir",
        "train_metadata", "validation_metadata", "seed", "epochs", "batch_size",
        "learning_rate", "warmup_epochs", "physical_gpu_id", "paddleclas",
    }
    absent = required - config.keys()
    if absent:
        raise ValueError(f"Missing configuration fields: {sorted(absent)}")
    if config["model_name"] != MODEL_NAME:
        raise ValueError("This entrypoint is for binary text-line orientation, not doc_ori.")
    if config["paddleclas"]["Arch"]["class_num"] != 2:
        raise ValueError("The training head must have exactly two classes (0 and 180).")
    for key in ("epochs", "batch_size"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if int(config["physical_gpu_id"]) < 0:
        raise ValueError("physical_gpu_id must be a nonnegative physical GPU index.")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive.")
    return config


def build_base_config(config: dict[str, Any], root: Path, output: Path) -> dict[str, Any]:
    """Make a complete PaddleClas recipe, replacing only portable runtime fields."""
    recipe = copy.deepcopy(config["paddleclas"])
    dataset = resolve_path(root, config["dataset_dir"])
    weights = resolve_path(root, config["pretrained_weights"])
    recipe["Global"].update({
        "seed": int(config["seed"]), "epochs": int(config["epochs"]),
        "device": "gpu", "output_dir": str(output),
        "pretrained_model": str(weights.with_suffix("")),
        "pdx_model_name": MODEL_NAME,
    })
    for split, filename, workers in (
        ("Train", "train.txt", config.get("train_workers", 16)),
        ("Eval", "val.txt", config.get("validation_workers", 8)),
    ):
        section = recipe["DataLoader"][split]
        section["dataset"].update({
            "image_root": str(dataset), "cls_label_path": str(dataset / filename),
        })
        section["loader"].update({
            "num_workers": int(workers),
            "use_shared_memory": bool(config.get("use_shared_memory", True)),
        })
    recipe["DataLoader"]["Train"]["sampler"]["batch_size"] = int(config["batch_size"])
    recipe["Optimizer"]["lr"].update({
        "learning_rate": float(config["learning_rate"]),
        "warmup_epoch": int(config["warmup_epochs"]),
    })
    recipe["Infer"]["PostProcess"]["class_id_map_file"] = str(dataset / "label.txt")
    return recipe


def child_environment(
    physical_gpu_id: int, cpu: bool = False,
    python_paths: tuple[Path, ...] = (),
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "" if cpu else str(physical_gpu_id)
    environment["FLAGS_cudnn_deterministic"] = "1"
    # PaddleClas uses imports such as `from ppcls...`. The completed server
    # experiment had a venv symlink; explicitly setting these roots removes
    # that hidden dependency while preserving any caller-supplied PYTHONPATH.
    if python_paths:
        prefixes = [str(path.resolve()) for path in python_paths]
        inherited = environment.get("PYTHONPATH", "")
        if inherited:
            prefixes.append(inherited)
        environment["PYTHONPATH"] = os.pathsep.join(prefixes)
    return environment


def build_command(
    stage: str, config: dict[str, Any], root: Path, output: Path,
    stage_output: Path, checkpoint: Path,
) -> list[str]:
    paddlex = resolve_path(root, config["paddlex_dir"])
    dataset = resolve_path(root, config["dataset_dir"])
    mode = "check_dataset" if stage == "check" else stage
    overrides = [
        f"Global.mode={mode}", f"Global.dataset_dir={dataset}",
        f"Global.output={stage_output}",
        # PaddleX rewrites CUDA_VISIBLE_DEVICES if passed gpu:0. Do not add :0.
        f"Global.device={'cpu' if stage == 'check' else 'gpu'}",
    ]
    if stage == "train":
        overrides.extend([
            "Train.num_classes=2", f"Train.epochs_iters={int(config['epochs'])}",
            f"Train.batch_size={int(config['batch_size'])}",
            f"Train.learning_rate={float(config['learning_rate'])}",
            # PaddleX maps warmup_steps to PaddleClas warmup_epoch.
            f"Train.warmup_steps={int(config['warmup_epochs'])}",
            f"Train.pretrain_weight_path={resolve_path(root, config['pretrained_weights'])}",
            f"Train.basic_config_path={output / 'paddleclas_seeded_config.yaml'}",
            "Train.eval_interval=1", "Train.save_interval=1",
        ])
    elif stage == "evaluate":
        overrides.append(f"Evaluate.weight_path={checkpoint}")
    elif stage == "export":
        overrides.append(f"Export.weight_path={checkpoint}")
    command = [sys.executable, str(paddlex / "main.py"), "-c", str(paddlex / MODEL_CONFIG_RELATIVE)]
    for override in overrides:
        command.extend(["-o", override])
    return command


def require_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}. "
            "Choose a new output directory for this stage."
        )


def verify_vendor(path: Path, expected_commit: str, name: str) -> str:
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"{name} must be a Git checkout at {path}: {error}") from error
    if actual != expected_commit:
        raise RuntimeError(f"{name} revision mismatch: expected {expected_commit}, got {actual}")
    return actual


def verify_manifests(dataset: Path) -> dict[str, Any]:
    """Read every path, enforce binary labels and balanced paired manifests."""
    labels_path = dataset / "label.txt"
    if labels_path.read_text(encoding="utf-8").splitlines() != ["0 0_degree", "1 180_degree"]:
        raise ValueError("label.txt must map 0 to 0_degree and 1 to 180_degree.")
    summary = {}
    split_paths = {}
    for split, name in (("train", "train.txt"), ("validation", "val.txt")):
        counts = {0: 0, 1: 0}
        paths = set()
        with (dataset / name).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                parts = line.strip().rsplit(maxsplit=1)
                if len(parts) != 2 or parts[1] not in {"0", "1"}:
                    raise ValueError(f"{name}:{number}: malformed path/label pair")
                relative, label = parts[0], int(parts[1])
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError(f"{name}:{number}: path must stay relative to dataset")
                path = dataset / relative
                if not path.is_file():
                    raise FileNotFoundError(f"{name}:{number}: missing image {path}")
                if relative in paths:
                    raise ValueError(f"{name}:{number}: duplicate image {relative}")
                paths.add(relative)
                counts[label] += 1
        if not counts[0] or counts[0] != counts[1]:
            raise ValueError(f"{name}: expected nonempty balanced 0/180 pairs, got {counts}")
        split_paths[split] = paths
        summary[split] = {"count": sum(counts.values()), "class_0": counts[0], "class_1": counts[1]}
    if split_paths["train"] & split_paths["validation"]:
        raise ValueError("Train and validation manifests share image paths.")
    return summary


def verify_group_split(train_csv: Path, validation_csv: Path) -> dict[str, int]:
    groups = []
    for path in (train_csv, validation_csv):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "group_id" not in (reader.fieldnames or []):
                raise ValueError(f"{path}: group_id is required to verify page-disjoint split")
            current = set()
            for row in reader:
                if not row["group_id"]:
                    raise ValueError(f"{path}: empty group_id")
                current.add(row["group_id"])
            groups.append(current)
    if groups[0] & groups[1]:
        raise ValueError("Train/validation share source_page groups; split before creating orientations.")
    return {"train_groups": len(groups[0]), "validation_groups": len(groups[1])}


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Called only after choosing CUDA visibility. No Paddle import at module scope."""
    if path.suffix != ".pdparams" or "doc_ori" in path.name:
        raise ValueError("Supply a trusted binary textline_ori .pdparams checkpoint, not doc_ori or ONNX.")
    import paddle

    state = paddle.load(str(path))
    weight = state.get("fc.weight")
    bias = state.get("fc.bias")
    if weight is None or bias is None or tuple(weight.shape) != (1280, 2) or tuple(bias.shape) != (2,):
        raise ValueError("Checkpoint must have PP-LCNet_x1_0 binary head fc.weight=(1280,2), fc.bias=(2,).")
    return {"paddle_version": paddle.__version__, "fc_weight_shape": list(weight.shape)}


def verify_gpu(physical_gpu_id: int, max_used_mb: int) -> str:
    # Check before importing Paddle so our own context does not inflate the busy check.
    command = [
        "nvidia-smi", f"--id={physical_gpu_id}",
        "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        status = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot inspect physical GPU {physical_gpu_id}: {error}") from error
    fields = [value.strip() for value in status.split(",")]
    if len(fields) != 6 or int(fields[0]) != physical_gpu_id:
        raise RuntimeError(f"Unexpected GPU status: {status}")
    if int(fields[3]) > max_used_mb:
        raise RuntimeError(f"Physical GPU {physical_gpu_id} already uses {fields[3]} MiB; not starting.")
    return status


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], root: Path, environment: dict[str, str], log: Path) -> None:
    with log.open("x", encoding="utf-8") as handle:
        with subprocess.Popen(
            command, cwd=root, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            return_code = process.wait()
    if return_code:
        raise RuntimeError(f"PaddleX exited with code {return_code}; full output: {log}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "train", "evaluate", "export"))
    parser.add_argument("--config", type=Path, default=Path("configs/finetune.yaml"))
    parser.add_argument("--root", type=Path, help="Repository/data root; default: config directory parent")
    parser.add_argument("--physical-gpu-id", type=int)
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Training experiment directory")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate/export this trusted .pdparams")
    parser.add_argument("--stage-output", type=Path, help="Fresh check/evaluation/export output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print only; no imports, GPU access, or writes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    root = args.root.resolve() if args.root else config_path.parent.parent
    config = load_config(config_path)
    if args.physical_gpu_id is not None:
        if args.physical_gpu_id < 0:
            raise ValueError("physical-gpu-id cannot be negative")
        config["physical_gpu_id"] = args.physical_gpu_id
    if args.pretrained_weights is not None:
        config["pretrained_weights"] = str(resolve_path(root, args.pretrained_weights))
    output = resolve_path(root, args.output_dir or config["output_dir"])
    checkpoint = resolve_path(root, args.checkpoint) if args.checkpoint else output / "best_model/best_model.pdparams"
    defaults = {
        "check": output.parent / (output.name + "_check"),
        "train": output,
        "evaluate": output / "evaluation",
        "export": output / "exported_inference",
    }
    if args.stage == "train" and args.stage_output is not None:
        raise ValueError("Use --output-dir for training, not --stage-output.")
    stage_output = resolve_path(root, args.stage_output) if args.stage_output else defaults[args.stage]
    environment = child_environment(
        int(config["physical_gpu_id"]), cpu=args.stage == "check",
        python_paths=(resolve_path(root, config["paddlex_dir"]),
                      resolve_path(root, config["paddleclas_dir"])),
    )
    command = build_command(args.stage, config, root, output, stage_output, checkpoint)
    print("CUDA_VISIBLE_DEVICES=" + environment["CUDA_VISIBLE_DEVICES"])
    print(shlex.join(command))
    if args.dry_run:
        print("Dry run: no files were written and no Paddle/GPU operation was performed.")
        return

    require_empty_output(stage_output)
    vendor_versions = {
        "paddlex_commit": verify_vendor(resolve_path(root, config["paddlex_dir"]), config["paddlex_commit"], "PaddleX"),
        "paddleclas_commit": verify_vendor(resolve_path(root, config["paddleclas_dir"]), config["paddleclas_commit"], "PaddleClas"),
    }
    paddleclas_dir = resolve_path(root, config["paddleclas_dir"])
    expected_paddleclas_dir = resolve_path(root, config["paddlex_dir"]) / "paddlex/repo_manager/repos/PaddleClas"
    if paddleclas_dir != expected_paddleclas_dir:
        raise ValueError(
            "Place the pinned PaddleClas checkout at " + str(expected_paddleclas_dir)
            + "; PaddleX's registry discovers this location."
        )
    if not (paddleclas_dir / ".installed").is_file():
        raise RuntimeError(
            "PaddleClas is cloned but not registered with PaddleX. After installing the "
            "training dependencies, run `python -m paddlex --install PaddleClas "
            "--use_local_repos --no_deps` using the pinned PaddleX checkout."
        )
    manifests = verify_manifests(resolve_path(root, config["dataset_dir"]))
    groups = verify_group_split(resolve_path(root, config["train_metadata"]), resolve_path(root, config["validation_metadata"]))
    model_config = resolve_path(root, config["paddlex_dir"]) / MODEL_CONFIG_RELATIVE
    if not model_config.is_file():
        raise FileNotFoundError(model_config)
    selected_weights = resolve_path(root, config["pretrained_weights"]) if args.stage in {"train", "check"} else checkpoint
    if not selected_weights.is_file():
        raise FileNotFoundError(selected_weights)
    gpu_status = None
    if args.stage != "check":
        gpu_status = verify_gpu(int(config["physical_gpu_id"]), int(config.get("max_existing_gpu_memory_mb", 1024)))
    # This process and all PaddleX children inherit the same physical GPU mapping.
    os.environ.update({key: environment[key] for key in ("CUDA_VISIBLE_DEVICES", "FLAGS_cudnn_deterministic")})
    checkpoint_info = inspect_checkpoint(selected_weights)
    if args.stage != "check":
        import paddle
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() != 1:
            raise RuntimeError("GPU training/evaluation/export requires a CUDA Paddle build and exactly one visible GPU.")

    stage_output.mkdir(parents=True, exist_ok=True)
    if args.stage == "train":
        with (output / "paddleclas_seeded_config.yaml").open("x", encoding="utf-8") as handle:
            yaml.safe_dump(build_base_config(config, root, output), handle, sort_keys=False, allow_unicode=True)
    run_info = {
        "stage": args.stage, "config": config, "root": str(root), "command": command,
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"], "physical_gpu": gpu_status,
        "pythonpath": environment.get("PYTHONPATH", ""),
        **vendor_versions, **checkpoint_info, "checkpoint": str(selected_weights),
        "checkpoint_sha256": sha256_file(selected_weights), "manifests": manifests, **groups,
        "python": sys.version, "seed_note": "Fixed seed; exact cross-hardware bitwise training is not guaranteed.",
    }
    with (stage_output / "run.json").open("x", encoding="utf-8") as handle:
        json.dump(run_info, handle, ensure_ascii=False, indent=2)
    run_logged(command, root, environment, stage_output / f"{args.stage}.log")


if __name__ == "__main__":
    main()
