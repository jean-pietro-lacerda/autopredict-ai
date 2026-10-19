import json
import os
from pathlib import Path
import re
import tempfile
import uuid

import joblib

if __package__:
    from .validation import project_path
else:
    from validation import project_path

MODEL_FILENAMES = ('random_forest_model.pkl', 'isolation_forest_model.pkl')


def _sync_directory(path):
    if os.name == 'posix':
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def load_models(models_dir='models'):
    root = project_path(models_dir)
    manifest = root / 'current.json'
    if manifest.exists():
        generation = json.loads(manifest.read_text())['generation']
        if not isinstance(generation, str) or not re.fullmatch(r'[0-9a-f]{32}', generation):
            raise ValueError('Referência de versão dos modelos inválida.')
        root = root / 'versions' / generation
    return tuple(joblib.load(root / filename) for filename in MODEL_FILENAMES)


def publish_models(classifier, detector, models_dir):
    root = project_path(models_dir)
    versions = root / 'versions'
    versions.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='.staging-', dir=versions) as staging:
        staging = Path(staging)
        for model, filename in zip((classifier, detector), MODEL_FILENAMES):
            target = staging / filename
            joblib.dump(model, target)
            joblib.load(target)
            with target.open('rb') as stream:
                os.fsync(stream.fileno())
        os.replace(staging, versions / generation)
        _sync_directory(versions)
    with tempfile.NamedTemporaryFile(mode='w', dir=root, prefix='.current-',
                                     delete=False) as stream:
        temporary_manifest = stream.name
        try:
            stream.write(json.dumps({'generation': generation}))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            os.unlink(temporary_manifest)
            raise
    try:
        os.replace(temporary_manifest, root / 'current.json')
        _sync_directory(root)
    finally:
        if os.path.exists(temporary_manifest):
            os.unlink(temporary_manifest)
