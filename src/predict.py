"""Evaluate and run a local exported PP-LCNet text-line orientation model.

Probabilities are aligned using per-image label_names. PaddleX 3.7.2 can return
the entire batch's class_ids matrix for each image; flattening that matrix and
zipping with two scores silently assigns the wrong probabilities.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


def probability_180(result):
    """Вернуть вероятность именно класса 180°, независимо от порядка top-k."""
    response = result.get('res', result)
    scores = np.asarray(response['scores'], dtype=float).reshape(-1)
    labels = list(response.get('label_names', []))
    # class_ids здесь не используем: в PaddleX 3.7.2 это может быть матрица
    # всего батча, а label_names и scores относятся к текущему изображению.
    if len(labels) != 2 or len(scores) != 2 or set(labels) != {'0_degree', '180_degree'}:
        raise ValueError('Expected two per-image scores and labels for 0/180 degrees')
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Invalid model probabilities')
    if not np.isclose(scores.sum(), 1.0, atol=2e-5):
        raise ValueError(f'Probabilities do not sum to one: {scores}')
    return float(scores[labels.index('180_degree')])


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def predict_paths(predictor, paths, batch_size, description):
    from tqdm.auto import tqdm
    probabilities = []
    for start in tqdm(range(0, len(paths), batch_size), desc=description):
        batch = paths[start:start + batch_size]
        # Нужны обе вероятности, а не только уверенность в победившем классе.
        # Порядок top-k меняется от картинки к картинке; его разберёт probability_180.
        results = list(predictor.predict(batch, batch_size=len(batch), topk=2))
        if len(results) != len(batch):
            raise RuntimeError('Prediction count does not match input count')
        for path, result in zip(batch, results):
            if Path(result['input_path']).resolve() != Path(path).resolve():
                raise RuntimeError('Predictor reordered input paths')
            probabilities.append(probability_180(result))
    return np.asarray(probabilities, dtype=np.float64)


def evaluate(predictor, manifest, root, metadata_path, output_dir, batch_size, limit):
    """Оценить вероятности на размеченном holdout, не на скрытом тесте."""
    records = [line.rsplit(' ', 1) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if limit is not None:
        records = records[:limit]
    paths = [str((root / path).resolve()) for path, _ in records]
    targets = np.asarray([int(label) for _, label in records])
    if not np.isin(targets, [0, 1]).all():
        raise ValueError('Validation manifest must contain only labels 0/1')
    if not all(Path(path).is_file() for path in paths):
        raise FileNotFoundError('Missing validation image')
    started = perf_counter()
    probabilities = predict_paths(predictor, paths, batch_size, 'Validation')
    elapsed = perf_counter() - started
    frame = pd.DataFrame({'image_path': paths, 'target': targets, 'p_180': probabilities})
    frame['language'] = frame['image_path'].map(lambda value: Path(value).stem.split('_')[0])
    if metadata_path:
        metadata = pd.read_csv(metadata_path)
        metadata['filename'] = metadata['image_path'].map(lambda value: Path(str(value).replace('\\', '/')).name)
        if metadata['filename'].duplicated().any():
            raise ValueError('Metadata filenames must be unique')
        types = metadata.set_index('filename')['content_type'].to_dict()
        frame['content_type'] = frame['image_path'].map(lambda value: types.get(Path(value).name, 'unknown'))
    # Brier оценивает исходные вероятности, accuracy — классы после порога 0.5.
    # Округление перед Brier потеряло бы информацию об уверенности модели.
    frame['squared_error'] = (frame['p_180'] - frame['target']) ** 2
    frame['correct'] = (frame['p_180'] >= .5).astype(int) == frame['target']
    metrics = {
        'validation_samples': len(frame),
        'accuracy': float(frame['correct'].mean()),
        'brier': float(frame['squared_error'].mean()),
        '1_minus_brier': 1 - float(frame['squared_error'].mean()),
        'mean_p180': float(frame['p_180'].mean()),
        'elapsed_seconds': elapsed,
        'images_per_second': len(frame) / elapsed,
        'probability_extraction': 'per-image label_names (class_ids batch matrix ignored)',
        'full_manifest': limit is None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'validation_predictions.csv', index=False)
    (output_dir / 'validation_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    groups = [('language', ['language']), ('target', ['target'])]
    if 'content_type' in frame:
        groups.append(('group', ['language', 'content_type']))
    for name, columns in groups:
        table = frame.groupby(columns, as_index=False).agg(
            count=('target', 'size'), accuracy=('correct', 'mean'),
            brier=('squared_error', 'mean'), mean_p180=('p_180', 'mean'),
        )
        table['1_minus_brier'] = 1 - table['brier']
        table.to_csv(output_dir / f'by_{name}.csv', index=False)
    print(json.dumps(metrics, indent=2), flush=True)


def make_submission(predictor, sample_path, images_dir, output, batch_size):
    """Сохранить p_180 с теми же идентификаторами и порядком, что в образце."""
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing submission: {output}')
    sample = pd.read_csv(sample_path, dtype={'image_id': str})
    if list(sample.columns) != ['image_id', 'p_180'] or sample['image_id'].duplicated().any():
        raise ValueError('Invalid sample_submission schema or duplicate IDs')
    paths = []
    for image_id in sample['image_id']:
        if Path(image_id).name != image_id:
            raise ValueError('image_id must be a filename')
        filename = image_id if Path(image_id).suffix else image_id + '.png'
        path = (images_dir / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(str(path))
    started = perf_counter()
    probabilities = predict_paths(predictor, paths, batch_size, 'Test')
    elapsed = perf_counter() - started
    # Значения p_180 из sample_submission — заглушки, а не известные ответы.
    # Поэтому используем только image_id и не вычисляем здесь тестовый Brier.
    submission = sample[['image_id']].copy()
    submission['p_180'] = probabilities
    assert len(submission) == len(sample) and np.isfinite(probabilities).all()
    assert submission['image_id'].tolist() == sample['image_id'].tolist()
    assert submission['p_180'].between(0, 1).all()
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False, float_format='%.9g')
    print(f'Saved {len(submission):,} rows: {output}; {len(submission)/elapsed:.1f} images/s', flush=True)
    return {'rows': len(submission), 'elapsed_seconds': elapsed, 'sha256': hash_file(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--physical-gpu-id', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--validation-manifest', type=Path)
    parser.add_argument('--validation-root', type=Path)
    parser.add_argument('--validation-metadata', type=Path)
    parser.add_argument('--validation-limit', type=int)
    parser.add_argument('--report-dir', type=Path, default=Path('verified_inference'))
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--output', type=Path, default=Path('submission_finetuned.csv'))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if bool(args.sample) != bool(args.images):
        parser.error('--sample and --images must be supplied together')
    if args.validation_manifest and not args.validation_root:
        parser.error('--validation-root is required with --validation-manifest')
    if not args.sample and not args.validation_manifest:
        parser.error('Supply test sample/images or validation manifest')
    # Устройство выбираем до импорта Paddle: выбранная физическая GPU станет gpu:0.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu_id) if args.device == 'gpu' else ''
    import paddlex
    model_dir = args.model_dir.resolve()
    for filename in ['inference.json', 'inference.pdiparams', 'inference.yml']:
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(model_dir / filename)
    # Resize, RGB-нормализация (ImageNet mean/std) и метки берутся из inference.yml.
    # Новые фильтры/ручной resize здесь не добавляем: вход должен совпадать с экспортом.
    predictor = paddlex.create_predictor(
        model_name='PP-LCNet_x1_0_textline_ori', model_dir=str(model_dir),
        device='gpu:0' if args.device == 'gpu' else 'cpu', topk=2,
    )
    if args.validation_manifest:
        evaluate(predictor, args.validation_manifest, args.validation_root,
                 args.validation_metadata, args.report_dir, args.batch_size, args.validation_limit)
    if args.sample:
        result = make_submission(predictor, args.sample, args.images, args.output, args.batch_size)
        result['model_sha256'] = {name: hash_file(model_dir / name) for name in [
            'inference.json', 'inference.pdiparams', 'inference.yml'
        ]}
        result['device'] = args.device
        result['batch_size'] = args.batch_size
        args.output.with_suffix('.run.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
