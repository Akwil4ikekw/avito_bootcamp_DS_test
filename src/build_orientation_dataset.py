"""Create portable, paired 0/180 PaddleX manifests from authoritative splits."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from prepare_data import root_path, source_image_path


def build_dataset(root: Path, processed_dir: Path, output_dir: Path, seed: int = 42) -> dict:
    """Создать две ориентации каждого кропа внутри уже заданного разбиения."""
    root, processed_dir, output_dir = root.resolve(), processed_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite or merge dataset: {output_dir}")
    frames = {"train": pd.read_csv(processed_dir / "train.csv"), "val": pd.read_csv(processed_dir / "validation.csv")}
    required = {"image_path", "language", "group_id"}
    sources = {}
    for split, frame in frames.items():
        if not required.issubset(frame.columns):
            raise ValueError(f"{split}: missing columns {sorted(required - set(frame.columns))}")
        paths = [source_image_path(root, value, processed_dir) for value in frame["image_path"]]
        if len(set(paths)) != len(paths):
            raise ValueError(f"{split}: duplicate base images")
        for path in paths:
            if any(char.isspace() for char in path.relative_to(processed_dir).as_posix()):
                raise ValueError("PaddleClas space-separated manifests require paths without whitespace.")
        sources[split] = paths
    # Сначала делим исходные страницы, и только потом создаём перевёрнутые копии.
    # Иначе одна строка или соседние строки страницы могли бы попасть в оба сплита.
    if set(frames["train"]["group_id"]) & set(frames["val"]["group_id"]):
        raise ValueError("Train/validation contain shared source pages.")
    if set(sources["train"]) & set(sources["val"]):
        raise ValueError("Train/validation contain shared base images.")
    output_dir.mkdir(parents=True)
    summary = {"seed": seed, "class_labels": {"0": "0_degree", "1": "180_degree"}, "splits": {}}
    for split, paths in sources.items():
        records = []
        for path in paths:
            relative = path.relative_to(processed_dir)
            upright = output_dir / "images" / "upright" / relative
            rotated = output_dir / "images" / "rot180" / split / relative
            upright.parent.mkdir(parents=True, exist_ok=True)
            rotated.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, upright)
            with Image.open(path) as image:
                # Точный поворот на 180° переставляет пиксели без интерполяции.
                # Исходный синтетический кроп считаем upright; метку 1 создаём сами.
                image.transpose(Image.Transpose.ROTATE_180).save(rotated, format="PNG", compress_level=3)
            records.extend([(upright.relative_to(output_dir).as_posix(), 0), (rotated.relative_to(output_dir).as_posix(), 1)])
        # Парное создание даёт точный баланс 50/50, а не случайное число классов.
        # Seed фиксирует порядок записей; сами метки остаются известными и неизменными.
        np.random.default_rng(seed if split == "train" else seed + 1).shuffle(records)
        (output_dir / f"{split}.txt").write_text("".join(f"{path} {label}\n" for path, label in records), encoding="utf-8")
        summary["splits"][split] = {"base_images": len(paths), "views": len(records), "class_0": len(paths), "class_1": len(paths)}
    (output_dir / "label.txt").write_text("0 0_degree\n1 180_degree\n", encoding="utf-8")
    (output_dir / "dataset_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--processed-dir", default="data/processed/line_crops")
    parser.add_argument("--output-dir", default="data/paddlex_textline_orientation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.root, root_path(args.root, args.processed_dir), root_path(args.root, args.output_dir), args.seed), indent=2))


if __name__ == "__main__":
    main()
