"""Small data-pipeline checks; never downloads source shards or runs training."""
from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_orientation_dataset import build_dataset
from prepare_data import METADATA_FIELDS, add_content_type, extract_line_crops, filter_metadata, root_path, split_metadata


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.processed = self.root / "data/processed/line_crops"
        self.processed.mkdir(parents=True)
        self.config = {
            "processed_dir": "data/processed/line_crops", "seed": 42,
            "min_aspect_ratio": 1.0, "validation_size": 0.15,
            "n_per_language": 2, "max_lines_per_page": 3,
            "min_crop_width": 10, "min_crop_height": 8, "png_compress_level": 3,
        }

    def tearDown(self):
        self.directory.cleanup()

    def make_data(self):
        rows = []
        for language in ("ru", "en"):
            for page in range(10):
                for line in range(2):
                    relative = Path(language) / f"{language}_{page:06d}_{line:03d}.png"
                    path = self.processed / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    array = np.zeros((8, 16, 3), dtype=np.uint8)
                    array[:3, :5] = [page + 1, line + 10, 255]
                    Image.fromarray(array).save(path)
                    rows.append({"image_path": path.relative_to(self.root).as_posix().replace("/", "\\"), "language": language, "source_page_id": page, "line_id": line, "text": "AБ2" if line else "text", "width": 16, "height": 8})
        # One excluded geometry per language, still present in diagnostics.
        for language in ("ru", "en"):
            rows.append({"image_path": f"data\\processed\\line_crops\\{language}\\excluded.png", "language": language, "source_page_id": 99, "line_id": 0, "text": "x", "width": 8, "height": 16})
        pd.DataFrame(rows).to_csv(self.processed / "metadata.csv", index=False)
        filter_metadata(self.config, self.root)
        split_metadata(self.config, self.root)

    def test_content_type_and_path_guard(self):
        classified = add_content_type(pd.DataFrame({"text": ["АБ", "abc123", "123", "!?", None]}))
        self.assertEqual(classified.content_type.tolist(), ["letters", "alphanumeric", "digits", "other", "other"])
        with self.assertRaises(ValueError):
            root_path(self.root, "../outside")

    def test_split_is_page_disjoint_and_outputs_are_not_overwritten(self):
        self.make_data()
        train = pd.read_csv(self.processed / "train.csv")
        val = pd.read_csv(self.processed / "validation.csv")
        self.assertFalse(set(train.group_id) & set(val.group_id))
        self.assertEqual(len(train) + len(val), 40)
        self.assertEqual(set(train.language), {"ru", "en"})
        with self.assertRaises(FileExistsError):
            split_metadata(self.config, self.root)
        with self.assertRaises(FileExistsError):
            filter_metadata(self.config, self.root)

    def test_pairing_and_deterministic_shuffle(self):
        self.make_data()
        first = self.root / "pairs1"
        second = self.root / "pairs2"
        summary = build_dataset(self.root, self.processed, first)
        build_dataset(self.root, self.processed, second)
        for split in ("train", "val"):
            self.assertEqual((first / f"{split}.txt").read_bytes(), (second / f"{split}.txt").read_bytes())
            records = [line.rsplit(" ", 1) for line in (first / f"{split}.txt").read_text().splitlines()]
            self.assertEqual(sum(label == "0" for _, label in records), sum(label == "1" for _, label in records))
            for name, label in records:
                path = first / name
                self.assertTrue(path.is_file())
                if label == "1":
                    base_relative = Path(name).relative_to(f"images/rot180/{split}")
                    with Image.open(path) as rotated, Image.open(self.processed / base_relative) as upright:
                        np.testing.assert_array_equal(np.asarray(rotated), np.rot90(np.asarray(upright), 2))
        self.assertEqual(summary["splits"]["train"]["views"] + summary["splits"]["val"]["views"], 80)
        self.assertEqual((first / "label.txt").read_text(), "0 0_degree\n1 180_degree\n")
        with self.assertRaises(FileExistsError):
            build_dataset(self.root, self.processed, first)

    def test_small_h5_extraction(self):
        h5_path = self.root / "fixture.h5"
        encoded = io.BytesIO()
        Image.new("RGB", (30, 20), color="white").save(encoded, format="PNG")
        with h5py.File(h5_path, "w") as file:
            images = file.create_dataset("images", (1,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            images[0] = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
            annotation = {"line_bboxes": [{"bbox": [0, 0, 12, 8], "text": "abc"}, {"bbox": [2, 8, 20, 10], "text": "A1"}]}
            file.create_dataset("annotations", data=[json.dumps(annotation)], dtype=h5py.string_dtype("utf-8"))
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        count = extract_line_crops(h5_path, "ru", self.config, self.root, writer)
        self.assertEqual(count, 2)
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual({r["text"] for r in rows}, {"abc", "A1"})
        self.assertTrue(all((self.root / r["image_path"]).is_file() for r in rows))
        with self.assertRaises(FileExistsError):
            extract_line_crops(h5_path, "ru", self.config, self.root, writer)

if __name__ == "__main__":
    unittest.main()
