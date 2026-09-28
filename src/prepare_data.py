"""Explicit, non-destructive stages for synthetic text-line preparation.

The saved crop PNGs and train/validation CSVs are the authoritative input
of the submitted experiment. The source revision was
recovered from both original shard download metadata files and is pinned in
configs/data.yaml. PNG bytes can still vary with dependency versions.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import yaml
from PIL import Image


METADATA_FIELDS = [
    "image_path", "language", "source_page_id", "line_id", "text", "width", "height"
]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The data config must be a YAML mapping.")
    for key in ("n_per_language", "max_lines_per_page", "min_crop_width", "min_crop_height"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if not 0 < float(config["validation_size"]) < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    return config


def root_path(root: Path, relative: str | Path) -> Path:
    """Resolve a relative path and refuse escapes from the explicit project root."""
    root = root.resolve()
    pure = PurePosixPath(str(relative).replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
        raise ValueError(f"Expected a project-relative path: {relative}")
    path = root.joinpath(*pure.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes project root: {relative}")
    return path


def source_image_path(root: Path, value: str, processed_dir: Path) -> Path:
    """Accept original Windows separators, or portable project/crop-relative paths."""
    value = str(value).replace("\\", "/")
    candidate = Path(value)
    if candidate.is_absolute():
        path = candidate.resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Image path is outside the project root: {value}")
    else:
        path = root_path(root, value)
        if not path.is_file():
            path = root_path(processed_dir, value)
    if not path.is_file() or not path.is_relative_to(processed_dir.resolve()):
        raise FileNotFoundError(f"Image is missing or outside processed_dir: {value}")
    return path


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(existing))


def download_shards(config: dict, root: Path) -> dict[str, Path]:
    # Imported only for the explicitly selected download stage.
    from huggingface_hub import HfApi, hf_hub_download

    raw_dir = root_path(root, config["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    filenames = {language: str(filename) for language, filename in config["files"].items()}
    for filename in filenames.values():
        root_path(raw_dir, filename)
    revision = HfApi().dataset_info(
        config["repo_id"], revision=config.get("revision") or "main"
    ).sha
    paths = {}
    for language, filename in filenames.items():
        existing = root_path(raw_dir, filename)
        if existing.is_file():
            print(f"{language}: reusing existing {existing}")
            paths[language] = existing
        else:
            paths[language] = Path(hf_hub_download(
                repo_id=config["repo_id"], filename=filename, repo_type="dataset",
                revision=revision, local_dir=str(raw_dir),
            ))
    provenance = raw_dir / "dataset_source.json"
    if not provenance.exists():
        provenance.write_text(json.dumps({
            "repo_id": config["repo_id"], "resolved_revision_for_this_download": revision,
            "files": filenames,
            "note": "The original revision was recovered from both HF shard download metadata files. Reused local H5 files are not rehashed here; verify their provenance before mixing sources.",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


def decode_annotation(value) -> dict:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def extract_line_crops(h5_path: Path, language: str, config: dict, root: Path, writer) -> int:
    import h5py
    from tqdm import tqdm

    output_dir = root_path(root, config["processed_dir"])
    language_dir = output_dir / language
    language_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))
    saved = 0
    with h5py.File(h5_path, "r") as h5_file:
        page_indices = rng.permutation(len(h5_file["images"]))
        for page_idx in tqdm(page_indices, desc=f"Extracting {language}"):
            image_bytes = h5_file["images"][page_idx]
            if hasattr(image_bytes, "tobytes"):
                image_bytes = image_bytes.tobytes()
            with Image.open(io.BytesIO(image_bytes)) as page_image:
                image = page_image.convert("RGB")
            annotation = decode_annotation(h5_file["annotations"][page_idx])
            lines = annotation["line_bboxes"]
            # Несколько строк с каждой случайно выбранной страницы увеличивают
            # разнообразие фонов и оформления при фиксированном числе кропов.
            line_indices = rng.permutation(len(lines))[:int(config["max_lines_per_page"])]
            for line_idx in line_indices:
                line = lines[int(line_idx)]
                x, y, width, height = map(int, line["bbox"])
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(image.width, x + width), min(image.height, y + height)
                if x2 - x1 < int(config["min_crop_width"]) or y2 - y1 < int(config["min_crop_height"]):
                    continue
                crop = image.crop((x1, y1, x2, y2))
                save_path = language_dir / f"{language}_{int(page_idx):06d}_{int(line_idx):03d}.png"
                if save_path.exists():
                    raise FileExistsError(f"Refusing to overwrite {save_path}")
                crop.save(save_path, format="PNG", compress_level=int(config["png_compress_level"]))
                writer.writerow({
                    "image_path": save_path.relative_to(root.resolve()).as_posix(),
                    "language": language, "source_page_id": int(page_idx),
                    "line_id": int(line_idx), "text": line.get("text", ""),
                    "width": crop.width, "height": crop.height,
                })
                saved += 1
                if saved >= int(config["n_per_language"]):
                    return saved
    return saved


def extract(config: dict, root: Path) -> Path:
    processed_dir = root_path(root, config["processed_dir"])
    metadata_path = processed_dir / "metadata.csv"
    refuse_existing([metadata_path] + [processed_dir / language for language in config["files"]])
    h5_paths = {language: root_path(root_path(root, config["raw_dir"]), filename)
                for language, filename in config["files"].items()}
    missing = [str(path) for path in h5_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Download the configured shards first: " + ", ".join(missing))
    processed_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted extraction intentionally leaves its files for diagnosis;
    # it is never silently resumed into inconsistent metadata.
    with metadata_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for language, h5_path in h5_paths.items():
            saved = extract_line_crops(h5_path, language, config, root, writer)
            print(f"{language}: {saved}/{config['n_per_language']}")
            if saved != int(config["n_per_language"]):
                raise RuntimeError(f"Insufficient valid crops in {h5_path}: {saved}")
    return metadata_path


def filter_metadata(config: dict, root: Path) -> Path:
    """Отфильтровать метаданные по геометрии, сохранив исходные PNG и отказы."""
    processed_dir = root_path(root, config["processed_dir"])
    filtered_path, rejected_path = (processed_dir / name for name in ("metadata_filtered.csv", "rejected_vertical.csv"))
    refuse_existing([filtered_path, rejected_path])
    frame = pd.read_csv(processed_dir / "metadata.csv")
    if ((frame["height"] <= 0) | (frame["width"] <= 0)).any():
        raise ValueError("Crop dimensions must be positive.")
    frame["aspect_ratio"] = frame["width"] / frame["height"]
    # width/height < 1 означает высокий узкий бокс, а не доказательство направления
    # текста. Это ограничение выбранной обучающей выборки, не детектор ориентации.
    reject = frame["aspect_ratio"] < float(config["min_aspect_ratio"])
    frame.loc[~reject].to_csv(filtered_path, index=False)
    frame.loc[reject].to_csv(rejected_path, index=False)
    print(f"Retained: {(~reject).sum()}, excluded by geometry: {reject.sum()}; PNGs were not deleted.")
    return filtered_path


def add_content_type(frame: pd.DataFrame) -> pd.DataFrame:
    """Добавить срезы для анализа ошибок; это не целевые метки ориентации."""
    frame = frame.copy()
    text = frame["text"].fillna("").astype(str)
    letters = text.str.contains(r"[A-Za-zА-Яа-яЁё]", regex=True)
    digits = text.str.contains(r"\d", regex=True)
    frame["content_type"] = np.select(
        [letters & digits, letters, digits], ["alphanumeric", "letters", "digits"], default="other"
    )
    return frame


def split_metadata(config: dict, root: Path) -> tuple[Path, Path]:
    """Сделать один воспроизводимый holdout без общих исходных страниц."""
    from sklearn.model_selection import GroupShuffleSplit

    processed_dir = root_path(root, config["processed_dir"])
    train_path, validation_path = processed_dir / "train.csv", processed_dir / "validation.csv"
    refuse_existing([train_path, validation_path])
    frame = add_content_type(pd.read_csv(processed_dir / "metadata_filtered.csv"))
    # Номера страниц повторяются в ru/en, поэтому идентификатор включает язык.
    # Все кропы одной страницы должны оставаться вместе: у них общие фон/шрифт.
    frame["group_id"] = frame["language"] + "_" + frame["source_page_id"].astype(str)
    train_parts, validation_parts = [], []
    for _, language_frame in frame.groupby("language"):
        # Делим отдельно каждый язык, чтобы оба были представлены в holdout.
        # test_size задаёт долю групп (страниц); доля строк может немного отличаться.
        splitter = GroupShuffleSplit(n_splits=1, test_size=float(config["validation_size"]), random_state=int(config["seed"]))
        train_index, val_index = next(splitter.split(language_frame, groups=language_frame["group_id"]))
        train_parts.append(language_frame.iloc[train_index])
        validation_parts.append(language_frame.iloc[val_index])
    train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=int(config["seed"])).reset_index(drop=True)
    validation = pd.concat(validation_parts, ignore_index=True).sample(frac=1, random_state=int(config["seed"])).reset_index(drop=True)
    if set(train["group_id"]) & set(validation["group_id"]):
        raise AssertionError("Source-page leakage between splits.")
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    print(f"Base images: train={len(train)}, validation={len(validation)}. One group-disjoint holdout, not k-fold CV.")
    return train_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download", "extract", "filter", "split"])
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config, root = load_config(args.config), args.root.resolve()
    {"download": download_shards, "extract": extract, "filter": filter_metadata, "split": split_metadata}[args.stage](config, root)


if __name__ == "__main__":
    main()
