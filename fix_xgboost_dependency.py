"""Add missing XGBoost and record an audited, reversible environment transition.

Run only after stopping other notebooks. Never changes source, fits, or scores.
"""
from pathlib import Path
import argparse
import copy
import datetime
import hashlib
import json
import os
import subprocess
import sys
import uuid

IDENTITY_KEYS = (
    'version', 'suite', 'config', 'cache_hash', 'data_metadata', 'source_hashes',
    'author_lock_sha256', 'environment', 'device', 'gpu', 'scope', 'evaluation',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def atomic(path, content):
    path = Path(path)
    temp = path.with_name(path.name + '.xgb-fix.tmp')
    temp.write_bytes(content)
    temp.replace(path)


def fresh_versions(project):
    code = (
        'import sys,json; sys.path.insert(0,sys.argv[1]); import gru; '
        'print(json.dumps(gru.research_versions()))'
    )
    return json.loads(subprocess.check_output(
        [sys.executable, '-c', code, str(project / 'src')], text=True,
        encoding='utf-8', cwd=project,
    ))


def validate_manifests(project, campaign, environment):
    """Preflight everything before changing any existing evidence file."""
    plan = read(campaign / 'campaign.json')
    if (campaign / 'evaluation/SEALED.json').exists():
        raise RuntimeError('Test already sealed. This repair is only for the pre-test missing-package failure.')
    for name, expected in plan['source'].items():
        if digest(project / 'src' / name) != expected:
            raise RuntimeError('Source changed; this repair must not bypass source checks: ' + name)
    manifests = []
    for suite in plan['suites']:
        root = campaign / 'suites' / suite
        path = root / 'manifest.json'
        if not path.exists():
            continue
        manifest = read(path)
        identity = {k: manifest[k] for k in IDENTITY_KEYS}
        if identity_hash(identity) != manifest['signature']:
            raise RuntimeError('Manifest signature is invalid: ' + str(path))
        old = manifest['environment']
        comparison = copy.deepcopy(old)
        comparison['packages']['xgboost'] = environment['packages']['xgboost']
        if comparison != environment:
            raise RuntimeError('Other environment changes detected; not an XGBoost-only repair: ' + suite)
        if old['packages'].get('xgboost') not in ('not installed', environment['packages']['xgboost']):
            raise RuntimeError('Refusing to replace an existing XGBoost version: ' + suite)
        if suite == 'xgboost' and old['packages'].get('xgboost') == 'not installed':
            if list((root / 'fits').glob('*/COMPLETE.json')):
                raise RuntimeError('XGBoost fits already exist despite missing-package manifest; inspect evidence first.')
        # Verify every retained full-proof hash before rebuilding its index.
        if (root / 'sha256.json').exists():
            for name, expected in read(root / 'sha256.json').items():
                if digest(root / name) != expected:
                    raise RuntimeError('Existing evidence hash mismatch: ' + str(root / name))
        if (root / 'FROZEN.json').exists():
            frozen = read(root / 'FROZEN.json')
            if frozen['manifest_sha256'] != digest(path):
                raise RuntimeError('Frozen manifest mismatch: ' + suite)
            for item in frozen['final']:
                record_path = root / 'fits' / item['name'] / 'COMPLETE.json'
                if digest(record_path) != item['record_sha256']:
                    raise RuntimeError('Frozen fit record changed: ' + item['name'])
                for name, expected in read(record_path)['hashes'].items():
                    if digest(record_path.parent / name) != expected:
                        raise RuntimeError('Frozen fit artifact changed: ' + name)
        manifests.append((root, manifest))
    return manifests


def migrate(project, campaign, environment):
    manifests = validate_manifests(project, campaign, environment)
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    backup = campaign / 'dependency_repairs' / tag
    changes = {}
    for root, original in manifests:
        if original['environment'] == environment:
            continue
        revised = copy.deepcopy(original)
        revised.setdefault('original_training_environment', copy.deepcopy(original['environment']))
        revised['environment'] = environment
        revised['signature'] = identity_hash({k: revised[k] for k in IDENTITY_KEYS})
        relative = Path('evidence/dependency_repairs') / tag
        event = {
            'kind': 'ADD_PREVIOUSLY_MISSING_XGBOOST_ONLY',
            'note': 'Environment field now describes resume environment. Earlier fits ran under original_training_environment. No fits/scores/config/source were changed.',
            'before': original['environment'], 'after': environment,
            'original_manifest_sha256': digest(root / 'manifest.json'),
            'backup': str(backup / 'suites' / root.name),
        }
        revised.setdefault('dependency_repairs', []).append(relative.as_posix() + '/transition.json')
        local = {
            root / relative / 'manifest_before.json': (root / 'manifest.json').read_bytes(),
            root / relative / 'transition.json': encode(event),
            root / 'manifest.json': encode(revised),
        }
        if (root / 'FROZEN.json').exists():
            frozen = read(root / 'FROZEN.json')
            local[root / relative / 'FROZEN_before.json'] = (root / 'FROZEN.json').read_bytes()
            frozen['manifest_sha256'] = hashlib.sha256(local[root / 'manifest.json']).hexdigest()
            local[root / 'FROZEN.json'] = encode(frozen)
        if (root / 'sha256.json').exists():
            hashes = read(root / 'sha256.json')
            for path, content in local.items():
                hashes[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
            local[root / 'sha256.json'] = encode(hashes)
        changes.update(local)
    if not changes:
        print('No manifest migration needed; environment already matches.')
        return None
    backup.mkdir(parents=True)
    journal = []
    originals = {}
    for path, content in changes.items():
        old = path.read_bytes() if path.exists() else None
        originals[path] = old
        if old is not None:
            target = backup / path.relative_to(campaign)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old)
        journal.append({'path': str(path), 'before_sha256': hashlib.sha256(old).hexdigest() if old is not None else None, 'after_sha256': hashlib.sha256(content).hexdigest()})
    atomic(backup / 'journal.json', encode({'state': 'prepared', 'changes': journal}))
    try:
        for path, content in changes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic(path, content)
        validate_manifests(project, campaign, environment)
    except BaseException:
        for path, old in originals.items():
            if old is None:
                path.unlink(missing_ok=True)
            else:
                atomic(path, old)
        atomic(backup / 'status.json', encode({'state': 'rolled_back'}))
        raise
    atomic(backup / 'status.json', encode({'state': 'completed', 'changes': len(changes)}))
    print('AUDITED_ENVIRONMENT_TRANSITION_COMPLETE:', backup)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--campaign', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    candidates = sorted((project / 'results/research_campaigns').glob('*/campaign.json'))
    if args.campaign:
        campaign = args.campaign.resolve()
    elif len(candidates) == 1:
        campaign = candidates[0].parent
    else:
        raise RuntimeError('Specify --campaign with the exact campaign directory:\n' + '\n'.join(str(p.parent) for p in candidates))
    print('Python:', sys.executable)
    print('Campaign:', campaign)
    environment_before = fresh_versions(project)
    validate_manifests(project, campaign, environment_before)
    cache = project / 'results/dependency_cache'
    for name in ['pip', 'temp']:
        (cache / name).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PIP_CACHE_DIR=str(cache / 'pip'), TEMP=str(cache / 'temp'), TMP=str(cache / 'temp'))
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', '--no-deps',
        'xgboost>=3.1,<4',
    ], check=True, env=env)
    subprocess.run([sys.executable, '-c', 'import xgboost; print("XGBoost:", xgboost.__version__)'], check=True)
    environment_after = fresh_versions(project)
    migrate(project, campaign, environment_after)
    print('READY: restart notebook15 kernel and Run All with the SAME campaign. Keep old failed attempts and proof ZIPs.')


if __name__ == '__main__':
    main()
