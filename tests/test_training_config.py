"""CPU-only unit tests: no Paddle, CUDA, external vendor, or training required."""
import csv
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("portable_train", ROOT / "src/train.py")
TRAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)


class TrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = TRAIN.load_config(ROOT / "configs/finetune.yaml")

    def test_completed_recipe_is_preserved(self):
        recipe = TRAIN.build_base_config(self.config, ROOT, ROOT / "experiments/test")
        self.assertEqual(recipe["Arch"]["class_num"], 2)
        self.assertEqual(recipe["Global"]["image_shape"], [3, 80, 160])
        self.assertEqual(recipe["Global"]["seed"], 42)
        self.assertEqual(recipe["Global"]["epochs"], 10)
        self.assertEqual(recipe["Optimizer"]["momentum"], 0.9)
        self.assertEqual(recipe["Optimizer"]["regularizer"]["coeff"], 1e-5)
        self.assertEqual(recipe["Optimizer"]["lr"]["learning_rate"], 0.05)
        self.assertEqual(recipe["Optimizer"]["lr"]["warmup_epoch"], 1)
        transforms = recipe["DataLoader"]["Train"]["dataset"]["transform_ops"]
        self.assertEqual([next(iter(item)) for item in transforms],
                         ["DecodeImage", "ResizeImage", "TimmAutoAugment", "NormalizeImage", "RandomErasing"])
        self.assertEqual(transforms[1]["ResizeImage"]["size"], [160, 80])
        self.assertEqual(transforms[2]["TimmAutoAugment"]["config_str"], "rand-m9-mstd0.5-inc1")
        self.assertEqual(recipe["DataLoader"]["Train"]["loader"]["num_workers"], 16)
        self.assertEqual(recipe["DataLoader"]["Eval"]["loader"]["num_workers"], 8)

    def test_gpu_mapping_is_not_overwritten_by_runner(self):
        command = TRAIN.build_command("train", self.config, ROOT, ROOT / "exp", ROOT / "exp", ROOT / "weights.pdparams")
        self.assertIn("Global.device=gpu", command)
        self.assertNotIn("Global.device=gpu:0", command)
        self.assertEqual(TRAIN.child_environment(5)["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(TRAIN.child_environment(7)["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(TRAIN.child_environment(5, cpu=True)["CUDA_VISIBLE_DEVICES"], "")

    def test_eval_and_export_use_explicit_checkpoint(self):
        weight = ROOT / "explicit.pdparams"
        for stage, override in (("evaluate", "Evaluate.weight_path="), ("export", "Export.weight_path=")):
            command = TRAIN.build_command(stage, self.config, ROOT, ROOT / "exp", ROOT / "target", weight)
            self.assertIn(override + str(weight), command)
            self.assertIn("Global.device=gpu", command)

    def test_vendor_pythonpath_preserves_inherited_paths(self):
        paddlex = ROOT / "vendor/PaddleX"
        paddleclas = paddlex / "paddlex/repo_manager/repos/PaddleClas"
        inherited = str(ROOT / "other-libraries")
        with patch.dict(os.environ, {"PYTHONPATH": inherited}):
            environment = TRAIN.child_environment(5, python_paths=(paddlex, paddleclas))
            self.assertEqual(environment["PYTHONPATH"].split(os.pathsep),
                             [str(paddlex.resolve()), str(paddleclas.resolve()), inherited])
            self.assertEqual(os.environ["PYTHONPATH"], inherited)

    def test_original_config_is_not_mutated(self):
        before = self.config["paddleclas"]["Global"].copy()
        TRAIN.build_base_config(self.config, ROOT, ROOT / "exp")
        self.assertEqual(self.config["paddleclas"]["Global"], before)

    def test_existing_models_are_protected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            TRAIN.require_empty_output(path)
            (path / "best_model.pdparams").touch()
            with self.assertRaises(FileExistsError):
                TRAIN.require_empty_output(path)

    def test_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            for filename in ("train.csv", "validation.csv"):
                with (path / filename).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["group_id"])
                    writer.writeheader()
                    writer.writerow({"group_id": "ru_000123"})
            with self.assertRaises(ValueError):
                TRAIN.verify_group_split(path / "train.csv", path / "validation.csv")


if __name__ == "__main__":
    unittest.main()
