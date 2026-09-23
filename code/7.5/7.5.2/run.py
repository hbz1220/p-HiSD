#!/usr/bin/env python3
# Section 7.5.2: Lane-Emden timing and memory experiments for Table 6(a,b).
# Compare index-1 HiSD and fixed H1 p-HiSD for p = 3, 5 on the prescribed grids.
# Save summary CSV files and measurement definitions to outputs/7.5/7.5.2/.

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import contextlib
import copy
import csv
import ctypes
import datetime as dt
import hashlib
import importlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile


OUTPUT_ROOT = (
    Path(__file__).resolve().parent
    / ".." / ".." /'..'/ "outputs" / "7.5" / "7.5.2"
).resolve()
MANUSCRIPT_GRIDS = (64, 128, 192, 256)
SOURCE_FILE = Path(__file__).resolve()
IMPORTED_SOURCE_SHA256 = hashlib.sha256(SOURCE_FILE.read_bytes()).hexdigest()
THREAD_ENV = {name: '1' for name in (
    'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}
THREAD_ENV.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',PYTHONDONTWRITEBYTECODE='1')
os.environ.update(THREAD_ENV)
sys.dont_write_bytecode = True
BOOT_START_NS = time.perf_counter_ns()
BOOT_START_UNIX = time.time()
BASE = ROOT = None


def _publication_data(value):
    import re
    def omitted(key):
        key = str(key).lower()
        return ('sha256' in key or 'sha_256' in key or 'sha-256' in key
                or key in {'code_hashes', 'code_hash', 'config_hash', 'initial_state_hashes',
                           'original_backup_path', 'original_backup_read_at_runtime',
                           'block_count_note', 'request_block_count'})
    if isinstance(value, dict):
        return {key: _publication_data(item) for key, item in value.items() if not omitted(key)}
    if isinstance(value, (list, tuple)):
        return [_publication_data(item) for item in value]
    if isinstance(value, str):
        def portable(match):
            path = match.group(0).replace('\\', '/')
            for marker in ('/outputs/', '/code/'):
                if marker in path:
                    return marker[1:] + path.rsplit(marker, 1)[1]
            return path.rstrip('/').rsplit('/', 1)[-1]
        value = re.sub(r"(?:/(?:Users|home|bicmr|private|tmp|opt|Applications)/|[A-Za-z]:[\\/](?:Users|Program Files)[\\/])[^\s\"'`<>;,]+", portable, value)
        value = re.sub(r'\b[0-9a-fA-F]{64}\b', '[omitted]', value)
        return value
    return value


def begin_session():
    global BASE, ROOT, _BOOTSTRAP_NEW_RUN_PATH, _BOOTSTRAP_NEW_RUN_TOKEN, _DIR_CREATED_NS
    global _SESSION_LOCK
    import tempfile
    import uuid
    import shutil
    name = hashlib.sha256(os.path.normcase(str(OUTPUT_ROOT)).encode()).hexdigest()[:16]
    lock = (Path(tempfile.gettempdir()) / ('section752-' + name + '.lock')).open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError('Another Section 7.5.2 experiment is running')
    _SESSION_LOCK = lock
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    directory = OUTPUT_ROOT / '.current_run'
    if directory.is_symlink():
        raise RuntimeError('Working directory must not be a symbolic link')
    if directory.exists():
        recovery = Path(tempfile.mkdtemp(prefix='section752-recovery-')) / 'previous_run'
        shutil.move(str(directory), str(recovery))
        print('Previous interrupted run preserved at ' + str(recovery), flush=True)
    directory.mkdir(exist_ok=False)
    BASE = ROOT = directory
    token = uuid.uuid4().hex
    _BOOTSTRAP_NEW_RUN_PATH = str(directory.resolve())
    _BOOTSTRAP_NEW_RUN_TOKEN = token
    _DIR_CREATED_NS = time.perf_counter_ns()
    os.environ['LANE_EMDEN_CURRENT_SESSION'] = str(directory)
    os.environ['LANE_EMDEN_SESSION_TOKEN'] = token
    return directory


_FINAL_RESULT_FILES = frozenset({
    'scalability_summary.csv', 'fixed_grid_cost_summary.csv',
    'cost_breakdown.csv', 'memory_structure.csv', 'RESULTS_OVERVIEW.md',
})
_ARCHIVE_RUN_OWNERS = {}


def initialize_archive_ownership(run_directory=None, *, created_current_run=False):
    import uuid
    raw = Path(run_directory or BASE)
    if raw.is_symlink():
        raise RuntimeError('Refusing ownership through a symlink')
    directory = raw.resolve()
    if not directory.is_dir() or directory in _ARCHIVE_RUN_OWNERS:
        raise RuntimeError('Output directory is missing or already registered')
    existing = list(directory.iterdir())
    if created_current_run and (
        str(directory) != globals().get('_BOOTSTRAP_NEW_RUN_PATH')
        or not globals().get('_BOOTSTRAP_NEW_RUN_TOKEN')
    ):
        raise RuntimeError('Missing exclusive current-run directory provenance')
    if existing and not created_current_run:
        raise RuntimeError('Refusing ownership of a preexisting nonempty directory')
    marker = directory / '.current_run_ownership.json'
    if marker.exists() or marker.is_symlink():
        raise RuntimeError('Refusing an existing ownership marker')
    st = directory.stat()
    owner = dict(token=uuid.uuid4().hex, directory=str(directory),
                 parent_pid=os.getpid(), device=st.st_dev, inode=st.st_ino)
    with marker.open('x', encoding='utf-8') as handle:
        json.dump(owner, handle)
    _ARCHIVE_RUN_OWNERS[directory] = owner
    return owner['token']


def _owned_output_directory(run_directory=None, run_token=None):
    raw = Path(run_directory or BASE)
    if raw.is_symlink():
        raise RuntimeError('Refusing cleanup through a symlink')
    directory = raw.resolve()
    owner = _ARCHIVE_RUN_OWNERS.get(directory)
    if not owner or owner['parent_pid'] != os.getpid():
        raise RuntimeError('Refusing cleanup without current-process ownership')
    if run_token is not None and run_token != owner['token']:
        raise RuntimeError('Current-run ownership token mismatch')
    st = directory.stat()
    if (st.st_dev, st.st_ino) != (owner['device'], owner['inode']):
        raise RuntimeError('Owned directory was replaced; nothing was cleaned')
    marker = directory / '.current_run_ownership.json'
    if marker.is_symlink() or not marker.is_file() or marker.stat().st_nlink != 1:
        raise RuntimeError('Unsafe or missing current-run ownership marker')
    if json.loads(marker.read_text(encoding='utf-8')) != owner:
        raise RuntimeError('Current-run ownership marker changed')
    return directory


def _archive_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _archive_inventory(directory, excluded=()):
    import stat
    directory = Path(directory).resolve()
    excluded = set(excluded)
    result = {}
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory).as_posix()
        if relative in excluded:
            continue
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise RuntimeError('Refusing linked output path: ' + relative)
        if stat.S_ISDIR(st.st_mode):
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise RuntimeError('Refusing nonregular or shared output file: ' + relative)
        if not path.resolve().is_relative_to(directory):
            raise RuntimeError('Output path escaped the current directory')
        result[relative] = dict(size_bytes=st.st_size, sha256=_archive_hash(path),
                                device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns)
    return result


def _run_failure_details(record):
    monitor = record.get('monitor') or {}
    resources = monitor.get('pre_run_resources') or {}
    contention = monitor.get('pre_run_contention') or {}
    reason = (monitor.get('stop_reason') or record.get('not_run_reason')
              or record.get('failure_reason') or record.get('solver_exception')
              or record.get('cert_exception') or monitor.get('exception')
              or (record.get('resource_decision') or {}).get('reason'))
    details = dict(status=record.get('overall_status'), reason=reason)
    if resources:
        details['resources'] = {key: resources.get(key) for key in (
            'physical_ram_bytes', 'available_ram_estimate_bytes', 'disk_free_bytes')}
        details['resources'].update(
            effective_rss_limit_bytes=monitor.get('effective_rss_limit_bytes'),
            minimum_free_disk_bytes=monitor.get('minimum_free_disk_bytes'))
    if contention:
        details['cpu_check'] = {key: contention.get(key) for key in (
            'blocked', 'persistent_other_compute_python_pids', 'persistent_system_busy',
            'wait_elapsed_s', 'checks')}
        details['cpu_check']['samples'] = [dict(
            system_busy_fraction_estimate=s.get('system_busy_fraction_estimate'),
            busy_processes=[dict(pid=p.get('pid'), name=Path(p.get('comm', '')).name,
                                 cpu_percent=p.get('pcpu'))
                            for p in s.get('busy_processes', [])])
            for s in contention.get('snapshots', [])]
    return details


def _output_failure_details(exc, stage):
    configuration = {}
    details = {}
    tb = exc.__traceback__
    while tb is not None:
        local = tb.tb_frame.f_locals
        for name in ('config', 'cfg', 'record', 'r'):
            candidate = local.get(name)
            if not isinstance(candidate, dict):
                continue
            if candidate.get('overall_status') and candidate.get('overall_status') != 'verified_index1':
                details = _run_failure_details(candidate)
            candidate = candidate.get('config', candidate)
            if isinstance(candidate, dict) and 'N' in candidate and 'p' in candidate:
                configuration = {key: candidate[key] for key in
                    ('run_id', 'method', 'N', 'p', 'repetition', 'timing_mode')
                    if key in candidate}
        tb = tb.tb_next
    return dict(stage=stage, configuration=configuration,
                type=type(exc).__name__, message=str(exc)[:6000],
                details=details,
                interrupted=isinstance(exc, KeyboardInterrupt))


def _write_failure_outputs(directory, failure=None):
    failure = dict(failure or dict(stage='final_validation', message='Run did not pass all required checks'))
    if not failure.get('details'):
        for path in sorted((directory / 'runs').glob('*.monitor.json')):
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if record.get('overall_status') and record['overall_status'] != 'verified_index1':
                failure['details'] = _run_failure_details(record)
                if not failure.get('configuration'):
                    config = record.get('config', record)
                    failure['configuration'] = {key: config[key] for key in
                        ('run_id', 'method', 'N', 'p', 'repetition', 'timing_mode') if key in config}
                break
    failure = _publication_data(failure)
    stage = str(failure.get('stage', 'final_validation'))
    configuration = failure.get('configuration') or {}
    message = str(failure.get('message') or failure.get('reason') or failure)[:6000]
    text = ('INCOMPLETE / BLOCKED\nStage: ' + stage + '\nConfiguration: '
            + json.dumps(configuration, ensure_ascii=False, default=str)
            + '\nReason: ' + message + '\n')
    if failure.get('details'):
        text += 'Details: ' + json.dumps(failure['details'], ensure_ascii=False,
                                        indent=2, default=str) + '\n'
    for name in ('RUN_FAILURE.txt', 'RESULTS_OVERVIEW.md'):
        path = directory / name
        if path.is_symlink() or (path.exists() and
                (not path.is_file() or path.stat().st_nlink != 1)):
            raise RuntimeError('Refusing to overwrite an unsafe output: ' + name)
    (directory / 'RUN_FAILURE.txt').write_text(text, encoding='utf-8')
    overview = directory / 'RESULTS_OVERVIEW.md'
    old = overview.read_text(encoding='utf-8') if overview.exists() else '# Lane–Emden scalability results\n'
    old = '\n'.join(
        'Execution: **BLOCKED**. Evidence: **INSUFFICIENT**.'
        if line.startswith('Execution: ') else line
        for line in old.splitlines()
    )
    old = old.replace('all_automatic_validation_pass=True', 'all_automatic_validation_pass=False')
    old = old.split('\n## Run incomplete\n', 1)[0]
    overview.write_text(old.rstrip() + '\n\n## Run incomplete\n\n'
        'all_automatic_validation_pass=False. Existing tables are partial results; '
        'the complete requested experiment has not passed.\n\n'
        + text, encoding='utf-8')


def archive_evidence(success, run_directory=None, run_token=None, failure=None):
    directory = _owned_output_directory(run_directory, run_token)
    _archive_inventory(directory)
    if success:
        missing = [name for name in sorted(_FINAL_RESULT_FILES) if not (directory/name).is_file()]
        if missing:
            raise RuntimeError('Successful run lacks key files: ' + ', '.join(missing))
        if (directory/'RUN_FAILURE.txt').exists():
            raise RuntimeError('A failure report cannot be silently cleared as success')
    else:
        _write_failure_outputs(directory, failure)
    retained = _FINAL_RESULT_FILES | (frozenset() if success else {'RUN_FAILURE.txt'})
    inventory = _archive_inventory(directory)
    retained_hashes = {name: entry['sha256'] for name, entry in inventory.items() if name in retained}
    removable = [name for name in inventory if name not in retained and name != '.current_run_ownership.json']
    if _archive_inventory(directory) != inventory:
        raise RuntimeError('Output files changed during cleanup preflight')
    _owned_output_directory(directory, run_token)
    for name in removable:
        path = directory/name
        old = inventory[name]
        st = path.lstat()
        if path.is_symlink() or (
            st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_nlink
        ) != (old['device'], old['inode'], old['size_bytes'], old['mtime_ns'], 1):
            raise RuntimeError('Output file changed during cleanup: ' + name)
        if _archive_hash(path) != old['sha256']:
            raise RuntimeError('Output bytes changed during cleanup: ' + name)
        path.unlink()
    directories = []
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise RuntimeError('A link appeared during cleanup')
        if path.is_dir():
            directories.append(path)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.rmdir()
    for name, digest in retained_hashes.items():
        if _archive_hash(directory/name) != digest:
            raise RuntimeError('A retained key file changed during cleanup: ' + name)
    marker = '.current_run_ownership.json'
    expected = set(retained_hashes) | {marker}
    if {path.name for path in directory.iterdir()} != expected:
        raise RuntimeError('Unexpected files appeared; no broad cleanup attempted')
    _owned_output_directory(directory, run_token)
    (directory/marker).unlink()
    _ARCHIVE_RUN_OWNERS.pop(directory)
    remaining = {path.name for path in directory.iterdir()}
    if remaining != set(retained_hashes) or (success and remaining != _FINAL_RESULT_FILES):
        raise RuntimeError('Final output file set mismatch')
    return dict(success=bool(success), cleaned=True, archive_created=False,
                remaining_files=sorted(remaining), retained_file_hashes_verified=True)


def transfer_final_outputs(owner, cleanup):
    directory, root = Path(BASE), Path(OUTPUT_ROOT)
    names = sorted(_FINAL_RESULT_FILES)
    moved_old, moved_new = [], []
    previous = directory / '.previous_results'
    try:
        st = directory.stat()
        if (not cleanup or not cleanup.get('success') or not cleanup.get('cleaned')
                or owner['parent_pid'] != os.getpid()
                or (st.st_dev, st.st_ino) != (owner['device'], owner['inode'])
                or directory.parent != root):
            raise RuntimeError('Final transfer requires completed current-run cleanup')
        if {p.name for p in directory.iterdir()} != _FINAL_RESULT_FILES:
            raise RuntimeError('Final transfer requires the five result files')
        for name in names + ['RUN_FAILURE.txt']:
            path = root / name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise RuntimeError('Result destination is not a regular file: ' + name)
        previous.mkdir()
        for name in names + ['RUN_FAILURE.txt']:
            path = root / name
            if path.exists():
                os.replace(path, previous / name)
                moved_old.append(name)
        for name in names:
            os.replace(directory / name, root / name)
            moved_new.append(name)
    except BaseException as exc:
        for name in reversed(moved_new):
            os.replace(root / name, directory / name)
        for name in reversed(moved_old):
            os.replace(previous / name, root / name)
        if previous.exists():
            previous.rmdir()
        message = _publication_data(type(exc).__name__ + ': ' + str(exc))
        (directory / 'RUN_FAILURE.txt').write_text(
            'INCOMPLETE / BLOCKED\nStage: final_output_transfer\nReason: ' + message + '\n',
            encoding='utf-8')
        return dict(success=False, result_directory=str(directory), error=message)
    warnings = []
    try:
        for name in moved_old:
            (previous / name).unlink()
        previous.rmdir()
        directory.rmdir()
    except OSError as exc:
        warnings.append(_publication_data(str(exc)))
    return dict(success=True, result_directory=str(root),
                files=[str(root / name) for name in names], cleanup_warnings=warnings)


def run_archive_safety_tests():
    import tempfile
    import shutil
    checks = {}
    skipped = {}
    with tempfile.TemporaryDirectory(prefix='output_safety_tests_', dir=BASE) as tmp:
        testroot = Path(tmp)
        history = testroot/'history'
        history.mkdir()
        (history/'old_raw.json').write_text('historical fixture\n', encoding='utf-8')
        historical_hash = _archive_hash(history/'old_raw.json')
        def fixture(name):
            directory = testroot/name
            directory.mkdir()
            token = initialize_archive_ownership(directory)
            for filename in _FINAL_RESULT_FILES:
                (directory/filename).write_text('fixture ' + filename + '\n', encoding='utf-8')
            (directory/'raw').mkdir()
            (directory/'raw'/'endpoint.npz').write_bytes(b'temporary endpoint fixture')
            (directory/'raw'/'trace.csv').write_text('time,rss\n1,2\n', encoding='utf-8')
            (directory/'test_report.json').write_text('{"PASS": true}\n', encoding='utf-8')
            return directory, token
        try:
            try:
                initialize_archive_ownership(history)
                checks['preexisting_tree_rejected'] = False
            except RuntimeError:
                checks['preexisting_tree_rejected'] = True
            try:
                archive_evidence(True, history)
                checks['unowned_history_cleanup_rejected'] = False
            except RuntimeError:
                checks['unowned_history_cleanup_rejected'] = True
            good, token = fixture('success')
            before = {name: _archive_hash(good/name) for name in _FINAL_RESULT_FILES}
            result = archive_evidence(True, good, token)
            checks['success_exact_five_files'] = result['cleaned'] and {p.name for p in good.iterdir()} == _FINAL_RESULT_FILES
            checks['key_file_bytes_unchanged'] = all(_archive_hash(good/name) == digest for name, digest in before.items())
            failed, token = fixture('failure')
            before = {name: _archive_hash(failed/name) for name in _FINAL_RESULT_FILES if name.endswith('.csv')}
            result = archive_evidence(False, failed, token, failure=dict(
                stage='fixture_validation', configuration={'N': 64, 'p': 3}, message='synthetic failure'))
            checks['failure_at_most_one_extra_file'] = {p.name for p in failed.iterdir()} == _FINAL_RESULT_FILES | {'RUN_FAILURE.txt'}
            checks['failure_explicit_and_partial'] = 'fixture_validation' in (failed/'RUN_FAILURE.txt').read_text(encoding='utf-8') and 'all_automatic_validation_pass=False' in (failed/'RESULTS_OVERVIEW.md').read_text(encoding='utf-8')
            checks['failure_csv_bytes_unchanged'] = all(_archive_hash(failed/name) == digest for name, digest in before.items())
            early = testroot/'early_failure'
            early.mkdir()
            token = initialize_archive_ownership(early)
            (early/'temporary').mkdir()
            (early/'temporary'/'partial.json').write_text('{}', encoding='utf-8')
            archive_evidence(False, early, token, failure=dict(stage='startup', message='synthetic startup failure'))
            checks['early_failure_only_overview_and_error'] = {p.name for p in early.iterdir()} == {'RESULTS_OVERVIEW.md', 'RUN_FAILURE.txt'}
            missing, token = fixture('missing_key')
            (missing/'memory_structure.csv').unlink()
            try:
                archive_evidence(True, missing, token)
                checks['missing_key_prevents_success_cleanup'] = False
            except RuntimeError:
                checks['missing_key_prevents_success_cleanup'] = (missing/'raw'/'endpoint.npz').exists()
            linked, token = fixture('symlink')
            try:
                (linked/'history_link').symlink_to(history, target_is_directory=True)
            except OSError as exc:
                if os.name != 'nt' or getattr(exc, 'winerror', None) != 1314:
                    raise
                skipped['symlink_refused_without_history_access'] = 'Windows account lacks symlink privilege (1314)'
            else:
                try:
                    archive_evidence(True, linked, token)
                    checks['symlink_refused_without_history_access'] = False
                except RuntimeError:
                    checks['symlink_refused_without_history_access'] = (linked/'raw'/'endpoint.npz').exists()
                (linked/'history_link').unlink()
            wrong, token = fixture('wrong_token')
            try:
                archive_evidence(True, wrong, 'wrong-token')
                checks['wrong_token_rejected'] = False
            except RuntimeError:
                checks['wrong_token_rejected'] = (wrong/'raw'/'endpoint.npz').exists()
            checks['historical_fixture_unchanged'] = _archive_hash(history/'old_raw.json') == historical_hash
            checks['no_archives_created'] = not any(p.suffix == '.zip' for p in testroot.rglob('*'))
        finally:
            for directory in list(_ARCHIVE_RUN_OWNERS):
                if directory.is_relative_to(testroot):
                    _ARCHIVE_RUN_OWNERS.pop(directory)
    report = dict(PASS=all(checks.values()), checks=checks, skipped=skipped,
                  description='Current-run temporary output fixtures only; no solver or archive executed.')
    (BASE/'ARCHIVE_SAFETY_TEST_REPORT.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report

if __name__ == '__main__':
    begin_session()
    print('Fresh result directory: '+str(BASE),flush=True)
elif __name__ == '__mp_main__':
    BASE = ROOT = Path(os.environ['LANE_EMDEN_CURRENT_SESSION']).resolve()

if BASE is not None:
    os.environ['MPLCONFIGDIR'] = str(BASE/'matplotlib_cache')
    os.environ['XDG_CACHE_HOME'] = str(BASE/'cache')
    os.environ['TMPDIR'] = str(BASE/'temporary')
    os.environ['TEMP'] = os.environ['TMP'] = str(BASE/'temporary')
    (BASE/'temporary').mkdir(exist_ok=True)


SYSTEM = platform.system()
psutil = None
REQUIRED_DEPENDENCIES = ['numpy', 'scipy', 'matplotlib'] + (['psutil>=5.9'] if SYSTEM != 'Darwin' else [])

try:
    if sys.version_info < (3,10):
        raise RuntimeError('Python >=3.10 is required')
    if SYSTEM not in ('Darwin', 'Linux', 'Windows'):
        raise RuntimeError('Supported operating systems: macOS, Linux, Windows')
    if SYSTEM in ('Darwin', 'Linux'):
        import resource
    if SYSTEM != 'Darwin':
        import psutil
        if tuple(int(x) for x in psutil.__version__.split('.')[:2]) < (5, 9):
            raise RuntimeError('psutil >=5.9 is required for Linux/Windows resource monitoring')
    import numpy as np
    import scipy
    import scipy.linalg as la
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    if __name__ != '__mp_main__':
        import matplotlib
        matplotlib.use('Agg')
except Exception as dependency_error:
    if BASE is not None:
        with (BASE/'DEPENDENCY_FAILURE.json').open('x',encoding='utf-8') as handle:
            json.dump({'status':'BLOCKED','python':sys.executable,'exception':str(dependency_error),
                       'required':['Python >=3.10 on macOS/Linux/Windows'] + REQUIRED_DEPENDENCIES,
                       'automatic_installation':False,'traceback':traceback.format_exc()},handle,indent=2)
        if __name__ == '__main__':
            try:
                initialize_archive_ownership(BASE, created_current_run=True)
                archive_evidence(False, failure=_output_failure_details(dependency_error, 'dependency_check'))
            except Exception as cleanup_error:
                print('Dependency failure cleanup could not finish safely: '+str(cleanup_error), file=sys.stderr)
    raise RuntimeError('Environment check failed. Required packages: ' + ', '.join(REQUIRED_DEPENDENCIES) + '. No packages were installed automatically. '+str(dependency_error)) from dependency_error


CLOCK = time.perf_counter_ns
PROTOCOL = dict(alpha=1.0, amp=1.2, eta_phisd=0.5, eta_standard=1e-4,
                tau=1e-4, J=5, tol=1e-6, max_phisd=2000000, max_standard=2000000,
                div_grad=1e8, div_E=1e12, eig_tol=1e-8, eig_maxiter=3000,
                eig_ncv=20, cert_k=6, cert_resid=1e-7, cert_orth=1e-8, delta_ind=1e-6)

def sha_bytes(x):
    return hashlib.sha256(x).hexdigest()

def code_hash():
    return sha_bytes(Path(__file__).read_bytes())

# Use repeatable, separate initial and certification eigensolver starts per configuration.
def configuration(N, p, method='pHiSD_H1', profile=True, **extra):
    ident = f'{method}_N{N}_p{p}'
    seed = lambda phase: int.from_bytes(hashlib.sha256((ident+':'+phase).encode()).digest()[:4], 'little')
    result = dict(N=N, p=p, method=method, profile=profile, run_config_id=ident,
                  seed_init=seed('init'), seed_cert=seed('cert'), protocol=PROTOCOL.copy(),
                  output_dir=str(BASE/'runs'))
    result.update(extra)
    return result

def clean_json(x):
    if isinstance(x, dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def write_json(path, obj):
    with open(path, 'x', encoding='utf-8') as f: json.dump(clean_json(obj), f, indent=2, allow_nan=False)

# Five-point Dirichlet operator for -Delta on (0, pi)^2; N counts interior nodes per axis.
def build_laplacian_2d(N):
    h = np.pi / (N+1)
    e = np.ones(N)
    T = sp.diags([-e,4*e,-e], [-1,0,1], shape=(N,N), format='csr')
    I = sp.eye(N,format='csr')
    S = sp.diags([-e,-e], [-1,1],shape=(N,N),format='csr')
    return ((sp.kron(I,T,format='csr')+sp.kron(S,I,format='csr'))/h**2).tocsr()

def make_u0(N):
    h=np.pi/(N+1); x=np.arange(1,N+1)*h
    X,Y=np.meshgrid(x,x,indexing='ij')
    return (1.2*np.sin(X)*np.sin(Y)).reshape(-1),h

# Count numerical operations and distinguish nested timings from exclusive costs.
class Meter:
    def __init__(self, profile=True):
        self.profile=profile; self.phase='problem'; self.counts=defaultdict(lambda:defaultdict(int)); self.times=defaultdict(int)
    def add(self, key, n=1): self.counts[self.phase][key]+=n
    @contextmanager
    def span(self,key,leaf=False):
        enabled = self.profile or not leaf
        t=CLOCK() if enabled else None
        try: yield
        finally:
            if enabled: self.times[key]+=CLOCK()-t
    def operation(self, kind, f, x):
        self.add('n_'+kind+'_calls')
        rhs=1 if x.ndim==1 else x.shape[1]
        self.add('n_'+kind+'_rhs', rhs)
        t=CLOCK() if self.profile and kind in ('M_apply','M_solve') else None
        try: return f(x)
        finally:
            if t is not None: self.times[self.phase+':T_'+kind]+=CLOCK()-t
    def operator(self,kind,f,n):
        return sla.LinearOperator((n,n),matvec=lambda x:self.operation(kind,f,x),
                                  matmat=lambda x:self.operation(kind,f,x),dtype=np.dtype('float64'))
    def H(self,u,A,p):
        self.add('n_H_assemblies'); return A-sp.diags(p*u**(p-1),0,format='csr')
    def residual_energy(self,u,A,p,h):
        self.add('n_grad_evals'); self.add('n_residual_evals'); self.add('n_energy_evals')
        self.add('n_A_matvec'); Au=A@u; g=Au-u**p
        # p-HiSD evaluates A@u separately for the energy.
        if self.method=='pHiSD_H1': self.add('n_A_matvec'); Au=A@u
        E=0.5*float(u@Au)-float(np.sum(u**(p+1)))/(p+1)
        return g, float(h*np.linalg.norm(g)), float(np.linalg.norm(g)), E
    def normal(self,v,M=None):
        self.add('n_orth_calls')
        with self.span('T_orth_inclusive',leaf=True):
            d=float(v@self.operation('M_apply',M.__matmul__,v)) if M is not None else float(v@v)
            if d<=0 or not np.isfinite(d): raise ValueError('nonpositive/nonfinite normalization')
            return v/np.sqrt(d)
    def counters(self, exclude=('cert','audit')):
        result=defaultdict(int)
        for phase,c in self.counts.items():
            if phase not in exclude:
                for k,v in c.items(): result[k]+=v
        defaults=['n_outer_updates','n_residual_evals','n_grad_evals','n_energy_evals','n_H_assemblies','n_HVP','n_A_matvec','n_M_apply_calls','n_M_solve_calls','n_M_solve_rhs','n_frame_inner_steps','n_orth_calls','n_factorizations','n_pc_updates']
        for k in defaults: result[k]+=0
        result['n_HVP']=result['n_HVP_rhs']
        return dict(result)

@contextmanager
def factorization_audit(meter):
    sl=importlib.import_module('scipy.sparse.linalg._dsolve.linsolve')
    superlu=importlib.import_module('scipy.sparse.linalg._dsolve._superlu')
    original=superlu.gstrf
    def audited(*a,**kw):
        meter.add('n_factorizations')
        return original(*a,**kw)
    superlu.gstrf=audited
    try: yield
    finally: superlu.gstrf=original


def timing_record(m, total_ns):
    ts=m.times
    sec=lambda key: ts.get(key,0)/1e9
    eig=sec('T_eig_init_inclusive')+sec('T_frame_updates_inclusive')
    result={k:v/1e9 for k,v in ts.items()}
    result.update(T_total=total_ns/1e9,T_pc_update=0.0,T_pc_setup_update=sec('T_pc_setup_initial'),
                  T_pc_setup_initial=sec('T_pc_setup_initial'),T_M_build=sec('T_M_build'),T_factor=sec('T_factor'),
                  T_eig_inclusive=eig,T_eig_init_inclusive=sec('T_eig_init_inclusive'),
                  T_frame_updates_inclusive=sec('T_frame_updates_inclusive'),T_outer_inclusive=sec('T_outer_inclusive'),
                  T_problem_setup=sec('T_problem_setup'))
    result['T_driver_other']=result['T_total']-sum(result[k] for k in ['T_problem_setup','T_pc_setup_initial','T_eig_init_inclusive','T_outer_inclusive'])
    if m.profile:
        apply=sum(v for k,v in ts.items() if k.endswith(':T_M_apply') and k.split(':')[0] not in ('cert','audit'))/1e9
        solve=sum(v for k,v in ts.items() if k.endswith(':T_M_solve') and k.split(':')[0] not in ('cert','audit'))/1e9
        inside=sum(ts.get(phase+':T_'+op,0) for phase in ('init_eig','outer_frame') for op in ('M_apply','M_solve'))/1e9
        result.update(T_M_apply=apply,T_M_solve=solve,T_pc_apply_solve=apply+solve,T_pc_inside_eig=inside,
                      T_eig_exclusive_pc=eig-inside,T_orth_inclusive=sec('T_orth_inclusive'))
        result['T_other']=result['T_total']-result['T_pc_setup_update']-apply-solve-result['T_eig_exclusive_pc']
    else:
        for k in ('T_M_apply','T_M_solve','T_pc_apply_solve','T_pc_inside_eig','T_eig_exclusive_pc','T_orth_inclusive','T_other'): result[k]=None
        if m.method=='standard_HiSD':
            for k in ('T_M_apply','T_M_solve','T_pc_apply_solve','T_pc_inside_eig'):result[k]=0.0
            result['T_eig_exclusive_pc']=eig
            result['T_other']=result['T_total']-eig
    return result

def solve(config, initial_frame=None):
    if config.get('protocol') != PROTOCOL: raise ValueError('Frozen protocol mismatch')
    if config['p'] not in (3,5) or config['method'] not in ('pHiSD_H1','standard_HiSD'): raise ValueError('Unsupported configuration')
    N,p,method=config['N'],config['p'],config['method']; n=N*N
    m=Meter(config.get('profile',True));m.method=method
    r=dict(config=config,code_hash=code_hash(),N=N,n=n,p=p,method=method,
           includes_operator_assembly=True,includes_certification=False,solver_status='setup_failed',
           cert_status='not_attempted',overall_status='setup_failed',phase_timing_ns={},
           options=dict(splu=dict(permc_spec='COLAMD',diag_pivot_thresh=1.0,options={'Equil':True}),
                        eigsh=dict(which='SA',sigma=None,tol=1e-8,maxiter=3000,ncv=20)),
           started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           solver_start_highwater_bytes=highwater_bytes())
    A=M=factor=u=v=v_initial=None; hist=[]; energies=[]
    t0=CLOCK()
    try:
        with factorization_audit(m):
            with m.span('T_problem_setup'):
                u,h=make_u0(N); A=build_laplacian_2d(N); H0=m.H(u,A,p)
            # Factor M = A + I once and reuse it throughout the p-HiSD solve.
            if method=='pHiSD_H1':
                m.phase='pc_setup'
                with m.span('T_pc_setup_initial'):
                    with m.span('T_M_build'): M=(A+sp.eye(n,format='csr')).tocsc()
                    with m.span('T_factor'): factor=sla.splu(M,permc_spec='COLAMD',diag_pivot_thresh=1.0,options={'Equil':True})
            r['solver_status']='initialization_failed';m.phase='init_eig'
            with m.span('T_eig_init_inclusive'):
                start=np.random.Generator(np.random.PCG64(config['seed_init'])).standard_normal(n)
                r['init_start_vector_sha256']=sha_bytes(start.tobytes())
                H_op=m.operator('HVP',H0.__matmul__,n)
                if initial_frame is None:
                    kw=dict(k=1,which='SA',sigma=None,tol=1e-8,maxiter=3000,ncv=20,v0=start,return_eigenvectors=True)
                    if M is not None: kw.update(M=m.operator('M_apply',M.__matmul__,n),Minv=m.operator('M_solve',factor.solve,n))
                    vals,vecs=sla.eigsh(H_op,**kw); v=m.normal(vecs[:,0],M); r['init_lambda']=float(vals[0])
                else: v=m.normal(initial_frame.copy(),M);r['init_lambda']=None
                v_initial=v.copy()
            r['solver_status']='max_iter';m.phase='outer_state'
            with m.span('T_outer_inclusive'):
                updates=0;max_updates=2000000
                while True:
                    m.phase='outer_state'
                    g,res,raw,E=m.residual_energy(u,A,p,h);hist.append(res);energies.append(E)
                    if not np.isfinite([res,raw,E]).all() or not np.isfinite(u).all() or raw>1e8 or abs(E)>1e12:
                        r['solver_status']='numerical_divergence';break
                    # Stop on h * ||g||; the ordinary-Hessian index is checked separately.
                    if res<1e-6: r['solver_status']='success';break
                    if updates>=max_updates:break
                    H=m.H(u,A,p)
                    m.phase='outer_frame'
                    with m.span('T_frame_updates_inclusive'):
                        # Five frame sweeps at the current state precede each reflected state step.
                        for j in range(5):
                            Hv=m.operation('HVP',H.__matmul__,v)
                            if M is not None:
                                w=m.operation('M_solve',factor.solve,Hv)
                                Mw=m.operation('M_apply',M.__matmul__,w)
                                v=m.normal(v-1e-4*(w-v*(v@Mw)),M)
                            else:v=m.normal(v-1e-4*(Hv-v*(v@Hv)))
                            m.add('n_frame_inner_steps')
                    m.phase='outer_state'
                    z=m.operation('M_solve',factor.solve,g) if M is not None else g
                    eta=0.5 if M is not None else 1e-4
                    u=u+eta*(-z+2*v*(v@g));updates+=1;m.add('n_outer_updates')
    except Exception as e:
        r['solver_exception']=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
        if r['solver_status']=='max_iter':r['solver_status']='numerical_divergence'
    finally:
        end_ns=CLOCK(); total_ns=end_ns-t0
        r['solver_end_highwater_bytes']=highwater_bytes()
        r['numerical_intervals_ns']={'solver':[t0,end_ns]}
    r.update(timings=timing_record(m,total_ns),phase_timing_ns=dict(m.times),counters=m.counters(),
             phase_counters={k:dict(v) for k,v in m.counts.items()},history_length=len(hist),legacy_iter=len(hist),
             final_residual=hist[-1] if hist else None,energy=energies[-1] if energies else None,
             overall_status=r['solver_status'])
    obj=dict(A=A,M=M,factor=factor,u_final=u,v_final=v,v_initial=v_initial,
             residual_history=np.array(hist),energy_history=np.array(energies),meter=m)
    return r,obj

def classify_index_certificate(eigenvalues, absolute, relative, orthogonality, eigensolver_converged):
    """Classify a full six-pair ordinary-Hessian numerical index check.

    Sign uncertainty is distinct from evidence that the index is not one.
    An accurate positive first eigenvalue proves index zero; an accurate
    negative second eigenvalue proves at least two negative eigenvalues.
    """
    vals=np.asarray(eigenvalues);a=np.asarray(absolute);b=np.asarray(relative)
    complete=bool(eigensolver_converged and vals.shape==(6,) and a.shape==(6,) and b.shape==(6,))
    finite=bool(complete and np.isfinite(vals).all() and np.isfinite(a).all() and np.isfinite(b).all()
                and np.isfinite(orthogonality) and np.all(a>=0) and np.all(b>=0) and np.all(np.diff(vals)>=0))
    accurate=bool(finite and np.all(b<=1e-7) and 0<=orthogonality<=1e-8)
    index_one=bool(accurate and vals[0]+10*a[0]<-1e-6 and vals[1]-10*a[1]>1e-6)
    clearly_wrong=bool(accurate and (vals[0]-10*a[0]>1e-6 or vals[1]+10*a[1]<-1e-6))
    resolved=bool(index_one or clearly_wrong)
    status='verified_index1' if index_one else ('wrong_index' if clearly_wrong else 'verification_unresolved')
    return dict(eigenpair_accuracy_pass=accurate,index_sign_resolved=resolved,
                cert_sign_check=index_one,cert_accuracy_check=accurate,cert_status=status)

# Recompute the endpoint residual and six ordinary-Hessian eigenpairs.
# Certification lies outside the solver timer and checks residuals, signs and orthogonality.
def certify(r,obj):
    m=obj['meter'];m.phase='cert';u=obj['u_final'];A=obj['A'];p=r['p'];n=r['n'];h=np.pi/(r['N']+1)
    eigvals=eigvecs=np.array([]); t0=CLOCK()
    r.update(endpoint_finite=False,endpoint_residual_pass=False,eigensolver_converged=False,
             eigenpair_accuracy_pass=False,index_sign_resolved=False,cert_sign_check=False,
             cert_accuracy_check=False,cert_status='verification_unresolved')
    try:
        r['endpoint_finite']=bool(u is not None and np.isfinite(u).all())
        if not r['endpoint_finite']:raise ValueError('nonfinite or missing endpoint')
        g,res,raw,E=m.residual_energy(u,A,p,h);r['fresh_final_residual']=res;r['fresh_energy']=E
        r['endpoint_residual_pass']=bool(np.isfinite(res) and res<1e-6)
        H=m.H(u,A,p);op=m.operator('HVP',H.__matmul__,n)
        start=np.random.Generator(np.random.PCG64(r['config']['seed_cert'])).standard_normal(n)
        r['cert_start_vector_sha256']=sha_bytes(start.tobytes())
        with m.span('T_cert_eig'):
            eigvals,eigvecs=sla.eigsh(op,k=6,which='SA',sigma=None,tol=1e-8,maxiter=3000,ncv=20,v0=start,return_eigenvectors=True)
        r['eigensolver_converged']=True
        order=np.argsort(eigvals);eigvals=eigvals[order];eigvecs=eigvecs[:,order]
        eigvecs=eigvecs/np.linalg.norm(eigvecs,axis=0)
        absolute=np.linalg.norm(m.operation('HVP',H.__matmul__,eigvecs)-eigvecs*eigvals,axis=0)
        relative=absolute/np.maximum(1,np.abs(eigvals));orth=float(np.linalg.norm(eigvecs.T@eigvecs-np.eye(6),2))
        r.update(eigenvalues=eigvals.tolist(),eigenpair_absolute_residuals=absolute.tolist(),
                 eigenpair_relative_residuals=relative.tolist(),eigenvector_orthogonality_error=orth,
                 numerical_index=int(np.sum(eigvals < -1e-6)),nnz_H_final=int(H.nnz))
        r.update(classify_index_certificate(eigvals,absolute,relative,orth,r['eigensolver_converged']))
    except Exception as e:
        r.update(cert_status='verification_unresolved',eigenpair_accuracy_pass=False,index_sign_resolved=False,
                 cert_sign_check=False,cert_accuracy_check=False)
        r['cert_exception']=dict(type=type(e).__name__,message=str(e))
        if isinstance(e,sla.ArpackNoConvergence):
            eigvals=e.eigenvalues; eigvecs=e.eigenvectors
            r['partial_eigenvalues']=eigvals.tolist() if eigvals is not None else []
    finally:
        # Success requires both solver convergence and every endpoint certification check.
        overall_verified=bool(r['solver_status']=='success' and r['endpoint_finite']
            and r['endpoint_residual_pass'] and r['eigensolver_converged']
            and r['eigenpair_accuracy_pass'] and r['index_sign_resolved']
            and r['cert_status']=='verified_index1')
        if overall_verified:r['overall_status']='verified_index1'
        elif r['solver_status']!='success':r['overall_status']=r['solver_status']
        elif not r['endpoint_residual_pass']:r['overall_status']='endpoint_residual_failed'
        else:r['overall_status']='wrong_index' if r['cert_status']=='wrong_index' else 'verification_unresolved'
        end_ns=CLOCK(); r['timings']['T_cert']=(end_ns-t0)/1e9
        r['cert_end_highwater_bytes']=highwater_bytes()
        r['numerical_intervals_ns']['cert']=[t0,end_ns]
    r['timings']['T_total_with_cert']=r['timings']['T_total']+r['timings']['T_cert']
    r['timings']['T_cert_eig']=m.times.get('T_cert_eig',0)/1e9
    r['cert_counters']=dict(m.counts['cert'])
    r['phase_counters']['cert']=r['cert_counters']
    r['phase_timing_ns']=dict(m.times)
    obj.update(eigenvalues=eigvals,eigenvectors=eigvecs)
    return r,obj

def structural_diagnostics(r,obj):
    m=obj['meter'];m.phase='audit';M=obj['M'];A=obj['A'];f=obj['factor'];v=obj['v_final']
    payload=lambda mat:int(mat.data.nbytes+mat.indices.nbytes+mat.indptr.nbytes)
    r['structure']={'nnz_A':int(A.nnz) if A is not None else None,'nnz_M':int(M.nnz) if M is not None else 0,
                    'nnz_H_final':r.get('nnz_H_final'),'sparse_A_payload_bytes':payload(A) if A is not None else None,
                    'sparse_M_payload_bytes':payload(M) if M is not None else 0}
    if f is not None:
        L,U=f.L,f.U;fn=int(L.nnz+U.nnz)
        r['structure'].update(nnz_L=int(L.nnz),nnz_U=int(U.nnz),F_n=fn,fill_ratio=fn/M.nnz,
                              factor_exported_array_payload_bytes=payload(L)+payload(U))
    else:r['structure'].update(nnz_L=0,nnz_U=0,F_n=0,fill_ratio=None,factor_exported_array_payload_bytes=0)
    if v is not None:
        r['frame_normalization_error']=abs(float(v@m.operation('M_apply',M.__matmul__,v))-1) if M is not None else abs(float(v@v)-1)
    r['audit_counters']=dict(m.counts['audit']);r['phase_counters']['audit']=r['audit_counters']
    r['postprocessing_highwater_bytes']=highwater_bytes()
    return r

# Mark solver and certification intervals so RSS samples cover numerical work only.
def worker(config,conn):
    baseline_rss=rss_bytes()
    conn.send({'event':'baseline','rss_bytes':baseline_rss,'ru_maxrss_bytes':highwater_bytes(),'perf_counter_ns':CLOCK()})
    r,obj=solve(config)
    conn.send({'event':'solver_end','record':clean_json(r),'perf_counter_ns':CLOCK()})
    if obj['u_final'] is not None and r['solver_status'] not in ('setup_failed','initialization_failed'):
        conn.send({'event':'cert_start','perf_counter_ns':CLOCK()})
        r,obj=certify(r,obj)
    else:
        r['timings'].update(T_cert=0.0,T_total_with_cert=r['timings']['T_total'],T_cert_eig=0.0)
    conn.send({'event':'cert_end','record':clean_json(r),'perf_counter_ns':CLOCK()})
    postprocessing_start=CLOCK()
    structural_diagnostics(r,obj)
    r['postprocessing_intervals_ns']={'structure_export':[postprocessing_start,CLOCK()]}
    output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=True)
    path=output/(config['run_id']+'.npz')
    if path.exists():raise FileExistsError(path)
    arrays={k:obj[k] for k in ('u_final','v_final','v_initial','residual_history','energy_history','eigenvalues','eigenvectors') if obj.get(k) is not None}
    save_start=CLOCK()
    np.savez_compressed(path,**arrays)
    r['postprocessing_intervals_ns']['array_save']=[save_start,CLOCK()]
    r['arrays_path']=str(path);r['arrays_sha256']=sha_bytes(path.read_bytes())
    r['python']=sys.version;r['numpy']=np.__version__;r['scipy']=scipy.__version__
    r['thread_environment']={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    conn.send({'event':'done','record':clean_json(r)})
    conn.close()


MAX_ACTIVE_WALL_S = 10800.0
SAMPLE_INTERVAL_S = 0.020
_CASE_LOCK = threading.Lock()


class ProcTaskInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "virtual_size", "resident_size", "total_user", "total_system",
        "threads_user", "threads_system")] + [(name, ctypes.c_int32) for name in (
        "policy", "faults", "pageins", "cow_faults", "messages_sent",
        "messages_received", "syscalls_mach", "syscalls_unix", "csw",
        "threadnum", "numrunning", "priority")]


_LIBPROC = None
_MACH_TIMEBASE = None


def mach_timebase() -> tuple[int, int]:
    global _MACH_TIMEBASE
    if _MACH_TIMEBASE is None:
        class Timebase(ctypes.Structure):
            _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]
        info = Timebase()
        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        if system.mach_timebase_info(ctypes.byref(info)) != 0 or not info.denom:
            raise RuntimeError("Unable to read mach absolute-time conversion")
        _MACH_TIMEBASE = (int(info.numer), int(info.denom))
    return _MACH_TIMEBASE


def task_metrics(pid: int | None = None) -> dict | None:
    global _LIBPROC
    pid = os.getpid() if pid is None else int(pid)
    if SYSTEM != 'Darwin':
        try:
            process = psutil.Process(pid)
            with process.oneshot():
                memory = process.memory_info()
                cpu = process.cpu_times()
                threads = process.num_threads()
            return dict(rss_bytes=int(memory.rss),
                        cpu_user_ns=round(cpu.user * 1e9),
                        cpu_system_ns=round(cpu.system * 1e9),
                        thread_count=int(threads), runnable_thread_count=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _LIBPROC.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int,
            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        _LIBPROC.proc_pidinfo.restype = ctypes.c_int
    info = ProcTaskInfo()
    got = _LIBPROC.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
    if got != ctypes.sizeof(info):
        return None
    numer, denom = mach_timebase()
    return {"rss_bytes": int(info.resident_size),
            "cpu_user_ns": int(info.total_user) * numer // denom,
            "cpu_system_ns": int(info.total_system) * numer // denom,
            "cpu_user_mach_ticks": int(info.total_user),
            "cpu_system_mach_ticks": int(info.total_system),
            "thread_count": int(info.threadnum), "runnable_thread_count": int(info.numrunning)}


def rss_bytes(pid: int | None = None) -> int | None:
    metrics = task_metrics(pid)
    return metrics["rss_bytes"] if metrics else None


def highwater_bytes() -> int:
    if SYSTEM == 'Windows':
        memory = psutil.Process().memory_info()
        peak = getattr(memory, 'peak_rss', None)
        return int(memory.peak_wset if peak is None else peak)
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak * 1024 if SYSTEM == 'Linux' else peak


def monitor_metadata() -> dict:
    if SYSTEM == 'Darwin':
        return dict(rss_source='libproc proc_pidinfo PROC_PIDTASKINFO resident_size, bytes',
                    ru_maxrss_unit='bytes (macOS getrusage)',
                    rss_monitor_struct_size=ctypes.sizeof(ProcTaskInfo),
                    cpu_mach_timebase_numer_denom=list(mach_timebase()),
                    cpu_counter_conversion='Mach absolute CPU ticks * timebase numer/denom to nanoseconds',
                    runnable_thread_count_available=True)
    return dict(rss_source='psutil Process.memory_info().rss, bytes' +
                (' (Windows WorkingSetSize)' if SYSTEM == 'Windows' else ' (Linux resident set)'),
                ru_maxrss_unit=('bytes (Windows PeakWorkingSetSize)' if SYSTEM == 'Windows'
                                else 'bytes (Linux getrusage ru_maxrss KiB multiplied by 1024)'),
                rss_monitor_struct_size=None, cpu_mach_timebase_numer_denom=None,
                cpu_counter_conversion='psutil cumulative process CPU seconds * 1e9, rounded; OS counter resolution applies',
                runnable_thread_count_available=False, psutil_version=psutil.__version__)


def _command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE,
                                   timeout=10).strip()


def _sysctl(name: str) -> str:
    return _command(["sysctl", "-n", name])


def resources_snapshot(output_dir: str | Path = None) -> dict:
    output_dir = ROOT if output_dir is None else output_dir
    if SYSTEM == 'Darwin':
        physical = int(_sysctl("hw.memsize"))
        vm = _command(["vm_stat"])
        page_size = int(re.search(r"page size of (\d+) bytes", vm).group(1))
        pages = {key.strip(): int(value) for key, value in
                 re.findall(r"^([^:\n]+):\s+(\d+)\.", vm, re.MULTILINE)}
        available = sum(pages.get(key, 0) for key in
                        ("Pages free", "Pages inactive", "Pages speculative")) * page_size
        estimator = 'vm_stat (free + inactive + speculative) * page_size; excludes compressed and purgeable double counting; estimate, not guaranteed allocatable memory'
    else:
        memory = psutil.virtual_memory()
        physical, available = int(memory.total), int(memory.available)
        page_size, pages = None, None
        estimator = 'psutil.virtual_memory().available; platform OS estimate, not guaranteed allocatable memory'
    available = min(physical, available)
    usage = shutil.disk_usage(output_dir)
    return {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "physical_ram_bytes": physical, "available_ram_estimate_bytes": available,
        "available_ram_estimator": estimator,
        "page_size_bytes": page_size, "vm_stat_pages": pages,
        "rss_limit_bytes": int(min(0.55 * physical, 0.75 * available)),
        "rss_limit_rule": "min(0.55*physical_RAM, 0.75*pre_run_available_estimate)",
        "disk_free_bytes": usage.free, "disk_total_bytes": usage.total,
    }


def environment_info() -> dict:
    result = resources_snapshot(ROOT)
    if SYSTEM == 'Darwin':
        cpu = _sysctl('machdep.cpu.brand_string')
        physical_cores, logical_cores = int(_sysctl('hw.physicalcpu')), int(_sysctl('hw.logicalcpu'))
    else:
        cpu = platform.processor() or platform.machine()
        physical_cores, logical_cores = psutil.cpu_count(logical=False), psutil.cpu_count()
    result.update({"platform": platform.platform(), "architecture": platform.machine(),
        "python": sys.version, "python_executable": sys.executable,
        "cpu": cpu, "physical_cores": physical_cores, "logical_cores": logical_cores,
        "thread_environment": dict(THREAD_ENV),
        **monitor_metadata(),
        "sample_interval_s_requested": SAMPLE_INTERVAL_S,
        "self_rss_bytes": rss_bytes(), "self_highwater_bytes": highwater_bytes(),
        "monitor_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return result


_PORTABLE_CPU_PREVIOUS = None


def _portable_cpu_counters() -> dict:
    processes = {}
    inaccessible = 0
    for process in psutil.process_iter(['pid', 'ppid', 'name', 'create_time', 'cpu_times']):
        info = process.info
        if info['cpu_times'] is None or info['create_time'] is None:
            inaccessible += 1
            continue
        cpu = info['cpu_times']
        processes[(info['pid'], info['create_time'])] = dict(
            pid=info['pid'], ppid=info['ppid'], comm=info['name'] or '',
            cpu_s=cpu.user + cpu.system)
    system = psutil.cpu_times()._asdict()
    if SYSTEM == 'Windows':
        total = sum(system.get(k, 0) for k in ('user', 'system', 'idle'))
    else:
        total = sum(v for k, v in system.items() if k not in ('guest', 'guest_nice'))
    idle = system.get('idle', 0) + system.get('iowait', 0)
    return dict(monotonic=time.monotonic(), total=total, busy=total-idle,
                processes=processes, inaccessible=inaccessible)


def _portable_cpu_snapshot(exclude_pids: set[int]) -> dict:
    global _PORTABLE_CPU_PREVIOUS
    if _PORTABLE_CPU_PREVIOUS is None:
        _PORTABLE_CPU_PREVIOUS = _portable_cpu_counters()
        time.sleep(0.1)
    before = _PORTABLE_CPU_PREVIOUS
    after = _portable_cpu_counters()
    _PORTABLE_CPU_PREVIOUS = after
    elapsed = max(after['monotonic'] - before['monotonic'], 1e-9)
    rows, aggregate, excluded_cpu = [], 0.0, 0.0
    for identity, current in after['processes'].items():
        previous = before['processes'].get(identity)
        if previous is None:
            continue
        delta = max(0.0, current['cpu_s'] - previous['cpu_s'])
        if current['pid'] in exclude_pids:
            excluded_cpu += delta
            continue
        pcpu = 100 * delta / elapsed
        aggregate += pcpu
        if 'python' in current['comm'].lower() or pcpu >= 20:
            rows.append({k: current[k] for k in ('pid', 'ppid', 'comm')} | {'pcpu': pcpu})
    total_delta = after['total'] - before['total']
    busy_delta = after['busy'] - before['busy'] - excluded_cpu
    busy = min(1.0, max(0.0, busy_delta / total_delta)) if total_delta > 0 else 0.0
    return dict(unix_time=time.time(), busy_processes=rows, aggregate_pcpu=aggregate,
                logical_cores=psutil.cpu_count() or 1, system_busy_fraction_estimate=busy,
                sample_window_s=elapsed, inaccessible_process_count=after['inaccessible'],
                source='psutil CPU counter deltas; system busy excludes known own-process CPU; inaccessible processes remain included in system total')


def _cpu_snapshot(exclude_pids: set[int]) -> dict:
    if SYSTEM != 'Darwin':
        return _portable_cpu_snapshot(exclude_pids)
    lines = _command(["ps", "-axo", "pid=,ppid=,pcpu=,comm="]).splitlines()
    rows = []
    total = 0.0
    for line in lines:
        parts = line.strip().split(None, 3)
        if len(parts) != 4:
            continue
        try:
            pid, ppid, pcpu = int(parts[0]), int(parts[1]), float(parts[2])
        except ValueError:
            continue
        if pid in exclude_pids:
            continue
        comm = Path(parts[3]).name
        total += pcpu
        if "python" in comm.lower() or pcpu >= 20:
            rows.append({"pid": pid, "ppid": ppid, "pcpu": pcpu, "comm": comm})
    logical = os.cpu_count() or 1
    return {"unix_time": time.time(), "busy_processes": rows,
            "aggregate_pcpu": total, "logical_cores": logical,
            "system_busy_fraction_estimate": total / (100 * logical),
            "source": "ps pcpu EWMA proxy, not instantaneous system CPU utilization"}


def contention_check(exclude_pids: set[int] | None = None,
                     samples: int = 3, interval_s: float = 1.0) -> dict:
    global _PORTABLE_CPU_PREVIOUS
    if SYSTEM != 'Darwin':
        _PORTABLE_CPU_PREVIOUS = None
    excluded = set(exclude_pids or ()) | {os.getpid()}
    snapshots = []
    for i in range(samples):
        snapshots.append(_cpu_snapshot(excluded))
        if i + 1 < samples:
            time.sleep(interval_s)
    python_sets = [{r["pid"] for r in s["busy_processes"]
                   if "python" in r["comm"].lower() and r["pcpu"] > 20}
                   for s in snapshots]
    persistent_python = sorted(set.intersection(*python_sets)) if python_sets else []
    system_busy = bool(snapshots and all(
        s["system_busy_fraction_estimate"] > 0.80 for s in snapshots))
    return {"blocked": bool(persistent_python or system_busy),
            "persistent_other_compute_python_pids": persistent_python,
            "persistent_system_busy": system_busy, "snapshots": snapshots,
            "policy": "3 pre-run samples 1s apart; other Python >20% CPU in all, or system CPU estimate >80% logical capacity in all, blocks new timing case; never kill other tasks"}


def wait_for_contention(deadline, max_wait_s=300.0, retry_interval_s=5.0):
    started = time.monotonic()
    stop = started + max(0.0, min(max_wait_s, deadline - time.time()))
    contention = contention_check()
    checks = 1
    last_notice = None
    while contention['blocked']:
        remaining = min(stop - time.monotonic(), deadline - time.time())
        if remaining <= 0:
            break
        if last_notice is None or time.monotonic() - last_notice >= 30:
            print('Waiting for external CPU activity to subside before timing; '
                  f'up to {remaining:.0f} s remain. Other busy Python PIDs: '
                  + str(contention['persistent_other_compute_python_pids']), flush=True)
            last_notice = time.monotonic()
        time.sleep(min(retry_interval_s, remaining))
        if time.monotonic() >= stop or time.time() >= deadline:
            break
        contention = contention_check()
        checks += 1
    contention.update(wait_elapsed_s=time.monotonic() - started, checks=checks)
    return contention


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_clean(value.item())
    return value


def _write_json_exclusive(path: Path, value: dict):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _inside_root(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError("All monitor outputs must remain inside " + str(ROOT))
    return resolved


def _budget_deadline(config: dict) -> tuple[float, dict]:
    budget_s = min(float(config.get("active_wall_budget_s", MAX_ACTIVE_WALL_S)), MAX_ACTIVE_WALL_S)
    if budget_s <= 0:
        raise ValueError("active_wall_budget_s must be positive")
    path = ROOT / "resource_budget.json"
    if path.exists():
        budget = json.loads(path.read_text(encoding='utf-8'))
    else:
        budget = {"start_unix": time.time(), "active_wall_budget_s": budget_s,
                  "semantics": "Conservative elapsed wall from first monitor case, including imports, checks, inter-case idle, and postprocessing; never automatically extends authorized 3h limit"}
        try:
            _write_json_exclusive(path, budget)
        except FileExistsError:
            budget = json.loads(path.read_text(encoding='utf-8'))
    deadline = float(budget["start_unix"]) + min(budget_s, float(budget["active_wall_budget_s"]))
    if config.get("budget_deadline_unix") is not None:
        deadline = min(deadline, float(config["budget_deadline_unix"]))
    return deadline, budget


def _worker_entry(config, conn):
    try:
        if config['code_sha256'] != IMPORTED_SOURCE_SHA256 or IMPORTED_SOURCE_SHA256 != code_hash():
            raise RuntimeError('Source changed since configuration freeze')
        worker(config, conn)
    except BaseException as exc:
        try:
            conn.send({"event": "worker_exception", "perf_counter_ns": time.perf_counter_ns(),
                "exception_type": type(exc).__name__, "exception": str(exc),
                "traceback": traceback.format_exc(), "rss_bytes": rss_bytes(),
                "ru_maxrss_bytes": highwater_bytes()})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        conn.close()


def run_case(config: dict) -> dict:
    config = dict(config)
    output = _inside_root(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    run_id = str(config["run_id"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id must be a safe basename")
    record_path = output / (run_id + ".monitor.json")
    sample_path = output / (run_id + ".rss.csv")
    if record_path.exists() or sample_path.exists():
        raise FileExistsError("Immutable run_id already exists: " + run_id)
    if not _CASE_LOCK.acquire(blocking=False):
        raise RuntimeError('Another numerical worker created by this process is active')
    samples = []
    metric_samples = {}
    events = []
    record = {"run_id": run_id, "run_config_id": config.get("run_config_id"),
        "N": config.get("N"), "p": config.get("p"), "method": config.get("method"),
        "profile": config.get("profile"), "config": config, "code_hash": code_hash()}
    monitor = {"sampling_interval_s_requested": SAMPLE_INTERVAL_S,
        **monitor_metadata(),
        "highwater_semantics": "cumulative process OS high-water includes startup/import; never subtract high-water values to infer phase peaks",
        "sampled_peak_limitation": "20ms requested sampling may miss short peaks; actual max gap reported; solver/certification endpoint events do not provide direct current RSS; sampled peaks use only observed in-interval RSS samples; numerical peaks exclude post-certification structure export/serialization",
        "thread_environment": dict(THREAD_ENV),
        "monitor_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "record_path": str(record_path), "rss_samples_path": str(sample_path)}
    process = None
    parent = None
    interrupted = False
    phase = "startup"
    phase_started_ns = time.perf_counter_ns()
    launched_ns = phase_started_ns
    boundaries = {}
    terminal_reason = None
    last_record = None
    try:
        deadline, budget = _budget_deadline(config)
        monitor["resource_budget"] = budget
        monitor["budget_deadline_unix"] = deadline
        contention = (wait_for_contention(deadline)
                      if config.get("enforce_contention", True) else contention_check())
        monitor["pre_run_contention"] = contention
        resources = resources_snapshot(output)
        monitor["pre_run_resources"] = resources
        minimum_disk = int(config.get("min_free_disk_bytes", max(512 * 2**20,
            int(config.get("N", 1))**2 * 8 * 12)))
        rss_limit = resources["rss_limit_bytes"]
        if config.get("rss_limit_bytes") is not None:
            rss_limit = min(rss_limit, int(config["rss_limit_bytes"]))
        monitor["effective_rss_limit_bytes"] = rss_limit
        monitor["minimum_free_disk_bytes"] = minimum_disk
        refuse = None
        if time.time() >= deadline:
            refuse = "experiment active-wall budget exhausted"
        elif resources["disk_free_bytes"] < minimum_disk:
            refuse = "insufficient free disk space before worker startup"
        elif rss_limit < 64 * 2**20:
            refuse = "available RAM yields unsafe worker RSS cap below 64 MiB"
        elif config.get("enforce_contention", True) and contention["blocked"]:
            refuse = ("external CPU activity remained above the timing threshold after "
                      f"{contention['wait_elapsed_s']:.1f} s; worker not started")
            terminal_reason = "not_run_contention"
        elif config.get("predicted_required_wall_s", 0) > deadline - time.time():
            refuse = "remaining authorized wall budget insufficient for predicted configuration"
        if refuse:
            terminal_reason = terminal_reason or "not_run_resource_limit"
            monitor["stop_reason"] = refuse
            record.update({"solver_status": terminal_reason, "cert_status": "not_run",
                "overall_status": terminal_reason})
        else:
            context = mp.get_context("spawn")
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=_worker_entry, args=(config, child),
                                      name="lane-emden-" + run_id)
            launched_ns = phase_started_ns = time.perf_counter_ns()
            process.start()
            child.close()
            monitor["worker_pid"] = process.pid
            monitor["worker_start_method"] = "spawn"
            monitor["launched_unix"] = time.time()
            next_cpu = time.monotonic() + 15
            during_cpu = []
            pipe_eof = False
            while True:
                sample_ns = time.perf_counter_ns()
                metrics = task_metrics(process.pid)
                current = metrics["rss_bytes"] if metrics else None
                if current is not None:
                    samples.append((sample_ns, current, "sample"))
                    metric_samples[sample_ns] = metrics
                while parent.poll():
                    try:
                        event = parent.recv()
                    except EOFError:
                        pipe_eof = True
                        break
                    receipt_ns = time.perf_counter_ns()
                    name = event.get("event")
                    event_ns = int(event.get("perf_counter_ns", receipt_ns))
                    event["monitor_received_ns"] = receipt_ns
                    event["boundary_timestamp_source"] = "worker" if "perf_counter_ns" in event else "parent receipt (includes pipe latency)"
                    events.append(event)
                    if event.get("record") is not None:
                        last_record = event["record"]
                    event_rss = event.get("rss_bytes", event.get("rss"))
                    if event_rss is not None:
                        samples.append((event_ns, int(event_rss), "event:" + str(name)))
                    if name == "baseline":
                        boundaries["solver_start"] = event_ns
                        monitor["baseline_rss_bytes"] = event_rss
                        monitor["baseline_highwater_bytes"] = event.get("ru_maxrss_bytes", event.get("highwater_bytes"))
                        phase, phase_started_ns = "solver", event_ns
                    elif name == "solver_end":
                        boundaries["solver_end"] = event_ns
                        monitor["solver_end_highwater_bytes"] = event.get("ru_maxrss_bytes")
                        boundaries["cert_start"] = event_ns
                        phase, phase_started_ns = "cert", event_ns
                    elif name == "cert_start":
                        boundaries["cert_start"] = event_ns
                        phase, phase_started_ns = "cert", event_ns
                    elif name == "cert_end":
                        boundaries["cert_end"] = event_ns
                        monitor["cert_end_highwater_bytes"] = event.get("ru_maxrss_bytes")
                        phase, phase_started_ns = "postprocessing", event_ns
                    elif name == "done":
                        boundaries["done"] = event_ns
                        phase = "done"
                    elif name == "worker_exception":
                        monitor["worker_exception"] = event
                        terminal_reason = "worker_exception"
                if phase == "done":
                    break
                elapsed_phase = (time.perf_counter_ns() - phase_started_ns) / 1e9
                phase_limit = {
                    "startup": min(120.0, float(config.get("startup_timeout_s", 120))),
                    "solver": min(900.0, float(config.get("solver_timeout_s", 900))),
                    "cert": min(900.0, float(config.get("cert_timeout_s", 900))),
                    "postprocessing": min(900.0, float(config.get("postprocessing_timeout_s", 120))),
                }[phase]
                if current is not None and current > rss_limit:
                    terminal_reason = "memory_limit"
                    monitor["stop_reason"] = "own worker RSS exceeded frozen pre-run cap"
                elif elapsed_phase > phase_limit:
                    terminal_reason = {"solver": "timeout_solver", "cert": "timeout_cert"}.get(phase, "timeout_" + phase)
                    monitor["stop_reason"] = "own worker exceeded phase wall cap"
                elif time.time() >= deadline:
                    terminal_reason = "timeout_cert" if phase == "cert" else "timeout_solver"
                    monitor["stop_reason"] = "experiment active-wall budget exhausted; censored"
                if terminal_reason:
                    monitor["censor_phase"] = phase
                    monitor["phase_wall_s_at_censor"] = elapsed_phase
                    break
                if not process.is_alive():
                    if not pipe_eof and parent.poll():
                        continue
                    terminal_reason = "worker_exited_without_done"
                    monitor["stop_reason"] = "worker exited without complete immutable result"
                    break
                if time.monotonic() >= next_cpu:
                    during_cpu.append(_cpu_snapshot({os.getpid(), process.pid}))
                    next_cpu = time.monotonic() + 15
                time.sleep(SAMPLE_INTERVAL_S)
            monitor["during_run_cpu_snapshots"] = during_cpu
            if terminal_reason and process.is_alive():
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
            process.join(timeout=5)
            monitor["worker_exitcode"] = process.exitcode
            monitor["worker_elapsed_wall_s"] = (time.perf_counter_ns() - launched_ns) / 1e9
            if last_record is not None:
                record.update(last_record)
            if terminal_reason:
                record["overall_status"] = terminal_reason
                if phase in ("startup", "solver"):
                    record["solver_status"] = terminal_reason
                    record.setdefault("cert_status", "not_run")
                elif phase == "cert":
                    record["cert_status"] = terminal_reason
    except KeyboardInterrupt:
        interrupted = True
        terminal_reason = "monitor_interrupted"
        monitor["stop_reason"] = "operator interrupted monitor; only its own child terminated"
        record["overall_status"] = terminal_reason
    except Exception as exc:
        terminal_reason = "monitor_failed"
        record["overall_status"] = terminal_reason
        monitor["exception"] = {"type": type(exc).__name__, "message": str(exc),
                                "traceback": traceback.format_exc()}
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        if parent is not None:
            parent.close()
        now_ns = time.perf_counter_ns()
        monitor["event_boundaries_ns"] = dict(boundaries)
        intervals = record.get("numerical_intervals_ns", {})
        if intervals.get("solver"):
            boundaries["solver_start"], boundaries["solver_end"] = map(int, intervals["solver"])
        if intervals.get("cert"):
            boundaries["cert_start"], boundaries["cert_end"] = map(int, intervals["cert"])
        elif "numerical_intervals_ns" in record and "cert_end" in boundaries:
            boundaries.pop("cert_start", None)
            boundaries.pop("cert_end", None)
        monitor["boundaries_ns"] = boundaries
        monitor["numerical_boundary_source"] = "worker perf_counter_ns timer intervals" if intervals else "worker events / parent receipt fallback; censored phase may end at watchdog termination"
        monitor["solver_end_highwater_bytes"] = record.get("solver_end_highwater_bytes", monitor.get("solver_end_highwater_bytes"))
        monitor["cert_end_highwater_bytes"] = record.get("cert_end_highwater_bytes", monitor.get("cert_end_highwater_bytes"))
        start = boundaries.get("solver_start")
        solver_end = boundaries.get("solver_end", now_ns if phase == "solver" else None)
        cert_start = boundaries.get("cert_start")
        cert_end = boundaries.get("cert_end", now_ns if phase == "cert" else None)
        ordered = sorted(samples)
        solver_samples = [rss for stamp, rss, source in ordered
                          if start is not None and solver_end is not None and start <= stamp <= solver_end]
        cert_samples = [rss for stamp, rss, source in ordered
                        if cert_start is not None and cert_end is not None and cert_start <= stamp <= cert_end]
        numerical_samples = solver_samples + cert_samples
        monitor["solver_sampled_peak_rss_bytes"] = max(solver_samples, default=None)
        monitor["cert_sampled_peak_rss_bytes"] = max(cert_samples, default=None)
        monitor["solver_plus_cert_sampled_peak_rss_bytes"] = max(numerical_samples, default=None)
        numerical_stamps = [stamp for stamp, rss, source in ordered if source == "sample" and
            ((start is not None and solver_end is not None and start <= stamp <= solver_end) or
             (cert_start is not None and cert_end is not None and cert_start <= stamp <= cert_end))]
        gaps = [(b-a)/1e9 for a,b in zip(numerical_stamps, numerical_stamps[1:])]
        monitor["actual_max_numerical_sample_gap_s"] = max(gaps, default=None)
        monitor["cpu_phase_samples"] = {}
        for label, begin, end in (("solver", start, solver_end), ("cert", cert_start, cert_end)):
            rows = [(stamp, metric_samples[stamp]) for stamp in sorted(metric_samples)
                    if begin is not None and end is not None and begin <= stamp <= end]
            if len(rows) >= 2:
                first_stamp, first = rows[0]
                last_stamp, last = rows[-1]
                elapsed_ns = last_stamp - first_stamp
                cpu_ns = sum(last[k] - first[k] for k in ("cpu_user_ns", "cpu_system_ns"))
                runnable = [m['runnable_thread_count'] for _, m in rows]
                runnable_available = all(value is not None for value in runnable)
                monitor["cpu_phase_samples"][label] = {
                    "sample_window_wall_s": elapsed_ns/1e9,
                    "sample_window_cpu_s": cpu_ns/1e9,
                    "cpu_seconds_per_wall_second": cpu_ns/elapsed_ns,
                    "max_os_thread_count": max(m["thread_count"] for _,m in rows),
                    "max_os_runnable_thread_count": max(runnable) if runnable_available else None,
                    "mean_os_runnable_thread_count": sum(runnable)/len(rows) if runnable_available else None,
                    "interpretation": "sample-window process CPU/wall ratio empirically checks CPU concurrency; thread_count includes sleeping/runtime threads; brief unsampled concurrency may be missed"}
        monitor["sample_count"] = len(ordered)
        monitor["numerical_sample_count"] = len(numerical_stamps)
        monitor["events"] = events
        monitor["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        record["monitor"] = monitor
        record["baseline_rss_bytes"] = monitor.get("baseline_rss_bytes")
        record["solver_peak_rss_bytes"] = monitor["solver_sampled_peak_rss_bytes"]
        record["full_peak_rss_bytes"] = monitor["solver_plus_cert_sampled_peak_rss_bytes"]
        try:
            with sample_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["perf_counter_ns", "rss_bytes", "source", "phase", "cpu_user_ns", "cpu_system_ns", "thread_count", "runnable_thread_count"])
                for stamp, rss, source in ordered:
                    sample_phase = "outside_numerical"
                    if start is not None and solver_end is not None and start <= stamp <= solver_end:
                        sample_phase = "solver"
                    elif cert_start is not None and cert_end is not None and cert_start <= stamp <= cert_end:
                        sample_phase = "cert"
                    met = metric_samples.get(stamp, {})
                    writer.writerow([stamp, rss, source, sample_phase] +
                        [met.get(key) for key in ("cpu_user_ns", "cpu_system_ns", "thread_count", "runnable_thread_count")])
            _write_json_exclusive(record_path, record)
        finally:
            _CASE_LOCK.release()
    if interrupted:
        raise KeyboardInterrupt
    return record


REGRESSION_OUT = None
REGRESSION_SOURCE_HASH = None
_REG_BASELINE_START = None
REGRESSION_TOLERANCES = {
    'operator_relative': 1e-11, 'solve_relative': 1e-11,
    'generalized_lambda_absolute': 1e-8, 'generalized_aligned_M_vector': 1e-6,
    'history_rtol': 1e-5, 'history_atol': 1e-10, 'state_relative': 1e-8,
    'equal_update_counts': True, 'timer_toggle_exact': True,
    'complex_step_gradient_HVP_relative': 1e-11, 'energy_directional_relative': 1e-8,
    'ordinary_dense_lambda_absolute': 1e-8, 'ordinary_dense_six_eigenvalue_absolute': 1e-8,
}


def _baseline_eigsh(*args, **kwargs):
    if _REG_BASELINE_START is None:
        raise RuntimeError('Fresh explicit baseline starting vector was not supplied')
    kwargs.update(v0=_REG_BASELINE_START.copy(), ncv=20)
    return sla.eigsh(*args, **kwargs)


def _baseline_build_laplacian_2d(N):
    h = np.pi / (N + 1)
    e = np.ones(N)
    T = sp.diags([-e, 4.0 * e, -e], [-1, 0, 1], shape=(N, N), format='csr')
    I = sp.eye(N, format='csr')
    S = sp.diags([-e, -e], [-1, 1], shape=(N, N), format='csr')
    return ((sp.kron(I, T, format='csr') + sp.kron(S, I, format='csr')) / h ** 2).tocsr()

def _baseline_make_u0(N, amp):
    h = np.pi / (N + 1)
    x = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(x, x, indexing='ij')
    return ((amp * np.sin(X) * np.sin(Y)).reshape(-1), h)

def _baseline_energy(u, A, p):
    return 0.5 * float(u @ (A @ u)) - 1.0 / (p + 1.0) * float(np.sum(u ** (p + 1)))

def _baseline_grad(u, A, p):
    return A @ u - u ** p

def _baseline_hess(u, A, p):
    return A - sp.diags(p * u ** (p - 1), offsets=0, format='csr')

def _baseline_residual_h(g, h):
    return float(h * np.linalg.norm(g))

def _baseline_normalize_l2(v):
    nrm = np.linalg.norm(v)
    if nrm <= 0:
        raise ValueError('normalize_l2 got non-positive norm')
    return v / nrm

def _baseline_normalize_M(v, M_dot):
    denom = float(v @ M_dot(v))
    if denom <= 0:
        raise ValueError(f'normalize_M got non-positive M-norm: {denom}')
    return v / np.sqrt(denom)

def _baseline_merge_notes(*notes):
    return ' | '.join((s.strip() for s in notes if isinstance(s, str) and s.strip()))

def _baseline_history_stats(hist):
    arr = np.asarray(hist, dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    init, fin = (float(arr[0]), float(arr[-1]))
    rel = float(fin / init) if np.isfinite(init) and init > 0 and np.isfinite(fin) else np.nan
    return (init, fin, rel)

def _baseline_initialize_standard_direction(u0, A, p, cfg):
    H0 = _baseline_hess(u0, A, p)
    init_lambda, note = (np.nan, '')
    t0 = time.perf_counter()
    try:
        vals, vecs = _baseline_eigsh(H0, k=1, which='SA', tol=cfg['eigsh_tol'], maxiter=cfg['eigsh_maxiter'])
        v0, init_lambda = (_baseline_normalize_l2(vecs[:, 0]), float(vals[0]))
    except Exception as ex:
        v0 = _baseline_normalize_l2(u0.copy())
        note = f'ordinary eigsh failed; used L2-normalized u0 ({type(ex).__name__}: {ex})'
    return {'v0': v0, 'init_lambda': init_lambda, 'init_eig_time': float(time.perf_counter() - t0), 'note': note}

def _baseline_setup_h1_preconditioner_and_direction(u0, A, p, alpha, cfg):
    n, note = (u0.size, '')
    t_setup = time.perf_counter()
    M = (A + alpha * sp.eye(n, format='csr')).tocsc()
    t_factor = time.perf_counter()
    factor = sla.splu(M)
    factor_time = time.perf_counter() - t_factor

    def solve_M(b):
        return factor.solve(b)

    def M_dot(v):
        return M @ v
    init_lambda = np.nan
    t_eig = time.perf_counter()
    try:
        vals, vecs = _baseline_eigsh(_baseline_hess(u0, A, p), M=M, k=1, which='SA', tol=cfg['eigsh_tol'], maxiter=cfg['eigsh_maxiter'])
        v0, init_lambda = (_baseline_normalize_M(vecs[:, 0], M_dot), float(vals[0]))
    except Exception as ex:
        v0 = _baseline_normalize_M(u0.copy(), M_dot)
        note = f'generalized eigsh failed; used M-normalized u0 ({type(ex).__name__}: {ex})'
    return {'M': M, 'solve_M': solve_M, 'M_dot': M_dot, 'v0': v0, 'init_lambda': init_lambda, 'note': note, 'setup_time': float(time.perf_counter() - t_setup), 'factor_time': float(factor_time), 'init_eig_time': float(time.perf_counter() - t_eig)}

def _baseline_run_standard_single_eta(u0, A, p, h, eta_x, tau_v, cfg, max_iter):
    u = u0.copy()
    grad_hist, E_hist = ([], [])
    t_setup = time.perf_counter()
    init = _baseline_initialize_standard_direction(u0, A, p, cfg)
    v = init['v0'].copy()
    setup_time = time.perf_counter() - t_setup
    status = 'max_iter'
    t_iter = time.perf_counter()
    for _ in range(max_iter):
        Au = A @ u
        g = Au - u ** p
        ng_raw = float(np.linalg.norm(g))
        ng = _baseline_residual_h(g, h)
        E = 0.5 * float(u @ Au) - 1.0 / (p + 1.0) * float(np.sum(u ** (p + 1)))
        grad_hist.append(ng)
        E_hist.append(float(E))
        if not np.isfinite(ng_raw) or not np.isfinite(ng) or (not np.isfinite(E)) or (not np.all(np.isfinite(u))):
            status = 'diverged'
            break
        if ng_raw > cfg['div_grad'] or abs(E) > cfg['div_E']:
            status = 'diverged'
            break
        if ng < cfg['tol']:
            status = 'success'
            break
        H = _baseline_hess(u, A, p)
        for _j in range(cfg['J']):
            Hv = H @ v
            v = _baseline_normalize_l2(v - tau_v * (Hv - v * (v @ Hv)))
        u = u + eta_x * (-g + 2.0 * v * (v @ g))
    iter_time = time.perf_counter() - t_iter
    return {'method': 'standard_HiSD', 'status': status, 'iter': len(grad_hist), 'eta_x': float(eta_x), 'tau_v': float(tau_v), 'J': int(cfg['J']), 'max_iter_budget': int(max_iter), 'u_final': u, 'grad_norm_history': np.array(grad_hist, dtype=float), 'energy_history': np.array(E_hist, dtype=float), 'setup_time': float(setup_time), 'iter_time': float(iter_time), 'time_total': float(setup_time + iter_time), 'factor_time': 0.0, 'init_eig_time': float(init['init_eig_time']), 'init_lambda': float(init['init_lambda']) if np.isfinite(init['init_lambda']) else np.nan, 'note': init['note']}

def _baseline_run_standard_fixed(u0, A, p, h, eta_x, tau_v, cfg, max_iter):
    return _baseline_run_standard_single_eta(u0, A, p, h, eta_x, tau_v, cfg, max_iter)

def _baseline_run_phisd_h1_single_eta(u0, A, p, h, eta_x, cfg, max_iter, setup, success_check_first):
    u = u0.copy()
    v = setup['v0'].copy()
    solve_M, M_dot = (setup['solve_M'], setup['M_dot'])
    grad_hist, E_hist = ([], [])
    status = 'max_iter'
    t_iter = time.perf_counter()
    for _ in range(max_iter):
        g = _baseline_grad(u, A, p)
        ng_raw = float(np.linalg.norm(g))
        ng = _baseline_residual_h(g, h)
        E = _baseline_energy(u, A, p)
        grad_hist.append(ng)
        E_hist.append(float(E))
        if success_check_first and ng < cfg['tol']:
            status = 'success'
            break
        if not np.isfinite(ng_raw) or not np.isfinite(ng) or (not np.isfinite(E)) or (not np.all(np.isfinite(u))):
            status = 'diverged'
            break
        if ng_raw > cfg['div_grad'] or abs(E) > cfg['div_E']:
            status = 'diverged'
            break
        if not success_check_first and ng < cfg['tol']:
            status = 'success'
            break
        H = _baseline_hess(u, A, p)
        for _j in range(cfg['J']):
            Hv = H @ v
            w = solve_M(Hv)
            v = _baseline_normalize_M(v - cfg['h1_tau'] * (w - v * (v @ M_dot(w))), M_dot)
        u = u + eta_x * (-solve_M(g) + 2.0 * v * (v @ g))
    return {'status': status, 'iter': len(grad_hist), 'eta_x': float(eta_x), 'tau_v': float(cfg['h1_tau']), 'J': int(cfg['J']), 'max_iter_budget': int(max_iter), 'u_final': u, 'grad_norm_history': np.array(grad_hist, dtype=float), 'energy_history': np.array(E_hist, dtype=float), 'iter_time': float(time.perf_counter() - t_iter)}

def _baseline_run_h1_fixed(u0, A, p, h, alpha, cfg, success_check_first, catch_setup_error):
    try:
        setup = _baseline_setup_h1_preconditioner_and_direction(u0, A, p, alpha, cfg)
    except Exception as ex:
        if not catch_setup_error:
            raise
        return {'status': 'setup_failed', 'iter': 0, 'eta_x': np.nan, 'tau_v': float(cfg['h1_tau']), 'J': int(cfg['J']), 'max_iter_budget': int(cfg['h1_max_iter']), 'u_final': u0.copy(), 'grad_norm_history': np.array([], dtype=float), 'energy_history': np.array([], dtype=float), 'setup_time': 0.0, 'iter_time': 0.0, 'time_total': 0.0, 'factor_time': 0.0, 'init_eig_time': 0.0, 'init_lambda': np.nan, 'note': f'setup_failed: {type(ex).__name__}: {ex}'}
    run = _baseline_run_phisd_h1_single_eta(u0, A, p, h, cfg['h1_eta'], cfg, cfg['h1_max_iter'], setup, success_check_first)
    run.update({'setup_time': float(setup['setup_time']), 'iter_time': float(run['iter_time']), 'time_total': float(setup['setup_time'] + run['iter_time']), 'factor_time': float(setup['factor_time']), 'init_eig_time': float(setup['init_eig_time']), 'init_lambda': float(setup['init_lambda']) if np.isfinite(setup['init_lambda']) else np.nan})
    run['note'] = _baseline_merge_notes(setup.get('note', ''))
    return run

def reg_load_baseline():
    common = dict(tol=1e-6, J=5, eigsh_tol=1e-8, eigsh_maxiter=3000,
                  index_k=6, index_neg_tol=-1e-8, div_grad=1e8, div_E=1e12)
    b_cfg = dict(common, h1_tau=1e-4, h1_eta=0.5, h1_max_iter=5000)
    return {
        'build_laplacian_2d': _baseline_build_laplacian_2d,
        'energy': _baseline_energy,
        'grad': _baseline_grad,
        'hess': _baseline_hess,
        'history_stats': _baseline_history_stats,
        'initialize_standard_direction': _baseline_initialize_standard_direction,
        'make_u0': _baseline_make_u0,
        'merge_notes': _baseline_merge_notes,
        'normalize_M': _baseline_normalize_M,
        'normalize_l2': _baseline_normalize_l2,
        'residual_h': _baseline_residual_h,
        'run_h1_fixed': _baseline_run_h1_fixed,
        'run_phisd_h1_single_eta': _baseline_run_phisd_h1_single_eta,
        'run_standard_fixed': _baseline_run_standard_fixed,
        'run_standard_single_eta': _baseline_run_standard_single_eta,
        'setup_h1_preconditioner_and_direction': _baseline_setup_h1_preconditioner_and_direction,
        'B_CFG': b_cfg, 'A_STD_CFG': common,
    }


def reg_baseline_start(namespace, config):
    global _REG_BASELINE_START
    _REG_BASELINE_START = np.random.Generator(np.random.PCG64(config['seed_init'])).standard_normal(config['N'] ** 2)
    return _REG_BASELINE_START.copy()


def reg_dump(name, data):
    write_json(REGRESSION_OUT / name, data)


def reg_source_unchanged():
    return REGRESSION_SOURCE_HASH == code_hash()


def reg_save_solver_snapshot(label, record, objects):
    reg_dump(label + '_record.json', record)
    arrays = {key: value for key, value in objects.items()
              if isinstance(value, np.ndarray)}
    if arrays:
        np.savez_compressed(REGRESSION_OUT / (label + '_arrays.npz'), **arrays)


@contextmanager
def reg_real_factor_audit():
    backend = importlib.import_module('scipy.sparse.linalg._dsolve._superlu')
    original = backend.gstrf
    count = {'gstrf_calls': 0}

    def wrapped(*args, **kwargs):
        count['gstrf_calls'] += 1
        return original(*args, **kwargs)
    backend.gstrf = wrapped
    try:
        yield count
    finally:
        backend.gstrf = original

def reg_relative(x, y):
    return float(np.linalg.norm(x - y) / max(1.0, np.linalg.norm(y)))

def reg_align(x, y, M=None):
    inner = float(x @ (M @ y)) if M is not None else float(x @ y)
    return x if inner >= 0 else -x

def reg_small_tests(baseline):
    N = 8
    p = 3
    cfg = configuration(N, p)
    A = build_laplacian_2d(N)
    B = baseline['build_laplacian_2d'](N)
    u, h = make_u0(N)
    ub, hb = baseline['make_u0'](N, 1.2)
    M = (A + sp.eye(N * N)).tocsc()
    H = baseline['hess'](u, A, p)
    rng = np.random.Generator(np.random.PCG64(78103))
    z = rng.standard_normal(N * N)
    expected = 5 * N * N - 4 * N
    operator_error = reg_relative(A @ z, B @ z)
    grad = baseline['grad'](u, A, p)
    analytic = H @ z
    complex_HVP = np.imag(baseline['grad'](u + 1j * 1e-20 * z, A, p)) / 1e-20
    eps = 1e-05
    derivative = (baseline['energy'](u + eps * z, A, p) - baseline['energy'](u - eps * z, A, p)) / (2 * eps)
    expected_derivative = float(grad @ z)
    factor = sla.splu(M)
    sol = factor.solve(z)
    meter = Meter(True)
    meter.method = 'pHiSD_H1'
    v = meter.normal(z, M)
    oldcfg = baseline['B_CFG'].copy()
    reg_baseline_start(baseline, cfg)
    with reg_real_factor_audit() as oldaudit:
        oldsetup = baseline['setup_h1_preconditioner_and_direction'](u, A, p, 1.0, oldcfg)
    with reg_real_factor_audit() as newaudit:
        rec, obj = solve(cfg)
    reg_save_solver_snapshot('small_N8_solver_snapshot', rec, obj)
    assert rec['solver_status'] == 'success', rec
    v_new = reg_align(obj['v_initial'], oldsetup['v0'], M)
    diff = v_new - oldsetup['v0']
    normdiff = float(np.sqrt(max(0.0, diff @ (M @ diff))))
    initdense = la.eigh(H.toarray(), M.toarray(), subset_by_index=[0, 0], check_finite=True)[0][0]
    ordinary_dense = la.eigh(H.toarray(), subset_by_index=[0, 5], check_finite=True)[0]
    start = np.random.Generator(np.random.PCG64(cfg['seed_cert'])).standard_normal(N * N)
    ordinary = sla.eigsh(H, k=6, which='SA', sigma=None, tol=1e-08, maxiter=3000, ncv=20, v0=start)[0]
    certify(rec, obj)
    final_H = baseline['hess'](obj['u_final'], A, p)
    final_dense = la.eigh(final_H.toarray(), subset_by_index=[0, 5], check_finite=True)[0]
    vals = np.array(rec['eigenvalues'])
    metrics = {'operator_relative_error': operator_error, 'u0_relative_error': reg_relative(u, ub), 'gradient_HVP_relative_error': reg_relative(complex_HVP, analytic), 'energy_directional_relative_error': abs(derivative - expected_derivative) / max(1.0, abs(expected_derivative)), 'M_solve_relative_residual': reg_relative(M @ sol, z), 'normalization_error': abs(float(v @ (M @ v)) - 1), 'generalized_lambda_difference': abs(rec['init_lambda'] - oldsetup['init_lambda']), 'generalized_aligned_M_vector_difference': normdiff, 'generalized_dense_lambda_difference': abs(rec['init_lambda'] - initdense), 'ordinary_dense_eigenvalue_max_difference': float(np.max(np.abs(np.sort(ordinary) - ordinary_dense))), 'cert_dense_eigenvalue_max_difference': float(np.max(np.abs(vals - final_dense))), 'baseline_actual_gstrf_calls': oldaudit['gstrf_calls'], 'new_actual_gstrf_calls': newaudit['gstrf_calls'], 'new_recorded_factorizations': rec['counters']['n_factorizations'], 'Minv_calls': rec['phase_counters']['init_eig'].get('n_M_solve_calls', 0), 'nnz_A': A.nnz, 'expected_nnz_A': expected, 'cert_status': rec['cert_status']}
    checks = {'operator': operator_error <= REGRESSION_TOLERANCES['operator_relative'], 'u0': np.array_equal(u, ub) and h == hb, 'sparse_structure': A.nnz == expected, 'gradient_HVP': metrics['gradient_HVP_relative_error'] <= REGRESSION_TOLERANCES['complex_step_gradient_HVP_relative'], 'energy_gradient': metrics['energy_directional_relative_error'] <= REGRESSION_TOLERANCES['energy_directional_relative'], 'M_solve': metrics['M_solve_relative_residual'] <= REGRESSION_TOLERANCES['solve_relative'], 'normalization': metrics['normalization_error'] <= 1e-11, 'generalized_eigenvalue': metrics['generalized_lambda_difference'] <= REGRESSION_TOLERANCES['generalized_lambda_absolute'], 'generalized_vector': normdiff <= REGRESSION_TOLERANCES['generalized_aligned_M_vector'], 'generalized_dense': metrics['generalized_dense_lambda_difference'] <= REGRESSION_TOLERANCES['generalized_lambda_absolute'], 'ordinary_dense': metrics['ordinary_dense_eigenvalue_max_difference'] <= REGRESSION_TOLERANCES['ordinary_dense_lambda_absolute'], 'cert_dense': metrics['cert_dense_eigenvalue_max_difference'] <= REGRESSION_TOLERANCES['ordinary_dense_six_eigenvalue_absolute'], 'baseline_two_factorizations': oldaudit['gstrf_calls'] == 2, 'new_one_factorization': newaudit['gstrf_calls'] == rec['counters']['n_factorizations'] == 1, 'Minv_called': metrics['Minv_calls'] > 0, 'full_ordinary_certificate': rec['cert_status'] == 'verified_index1'}
    result = {'name': 'small_N8_tests', 'metrics': metrics, 'checks': checks, 'PASS': bool(all(checks.values()))}
    reg_dump('small_N8_tests.json', result)
    reg_dump('small_N8_driver_record.json', rec)
    np.savez_compressed(REGRESSION_OUT / 'small_N8_arrays.npz', u_final=obj['u_final'], v_initial=obj['v_initial'], eigenvalues=obj['eigenvalues'], eigenvectors=obj['eigenvectors'])
    return result

def reg_certification_adversarial_tests():
    cfg = configuration(8, 3)
    reference, refobj = solve(cfg)
    reg_save_solver_snapshot('adversarial_reference_solver_snapshot', reference, refobj)
    assert reference['solver_status'] == 'success'
    perturbed = refobj['u_final'].copy()
    perturbed[4 * 8 + 4] += 0.0001

    def execute_cert(u, A):
        meter = Meter(True)
        meter.method = 'pHiSD_H1'
        rec = dict(config=cfg, N=8, n=64, p=3, method='pHiSD_H1', solver_status='success', overall_status='success', timings={'T_total': 0.0}, numerical_intervals_ns={}, phase_counters={}, code_hash=code_hash(), purpose='adversarial_regression_fixture')
        obj = dict(meter=meter, u_final=u.copy(), A=A)
        certify(rec, obj)
        return (rec, obj)
    bad, badobj = execute_cert(perturbed, refobj['A'])
    badchecks = dict(perturbed_actual_endpoint=not np.array_equal(perturbed, refobj['u_final']), fresh_residual_fails=bad['fresh_final_residual'] > 1e-06, spectral_index_one=bad['cert_status'] == 'verified_index1', accuracy_pass=bad['eigenpair_accuracy_pass'], endpoint_residual_flag_false=not bad['endpoint_residual_pass'], overall_not_verified=bad['overall_status'] == 'endpoint_residual_failed')
    badresult = dict(name='adversarial_bad_endpoint_residual', checks=badchecks, PASS=all(badchecks.values()), fresh_final_residual=bad['fresh_final_residual'], cert_status=bad['cert_status'], overall_status=bad['overall_status'], eigenvalues=bad['eigenvalues'], eigenpair_relative_residuals=bad['eigenpair_relative_residuals'])
    reg_dump('adversarial_bad_endpoint_record.json', bad)
    np.savez_compressed(REGRESSION_OUT / 'adversarial_bad_endpoint_arrays.npz', u_reference=refobj['u_final'], u_perturbed=perturbed, eigenvalues=badobj['eigenvalues'], eigenvectors=badobj['eigenvectors'])
    diagonal = np.r_[-1.0, 5e-7, np.arange(1.0, 63.0)]
    H = sp.diags(diagonal, format='csr')
    zero = np.zeros(64)
    near, nearobj = execute_cert(zero, H)
    nearchecks = dict(full_symmetric_64_dimensional=H.shape == (64, 64) and (H - H.T).nnz == 0, actual_eigensolver_completed=near['eigensolver_converged'], accuracy_pass=near['eigenpair_accuracy_pass'], lambda2_near_delta=abs(near['eigenvalues'][1] - 5e-7) < 1e-10, sign_unresolved=not near['index_sign_resolved'], cert_unresolved=near['cert_status'] == 'verification_unresolved', overall_unresolved=near['overall_status'] == 'verification_unresolved')
    nearresult = dict(name='adversarial_near_zero_unresolved', checks=nearchecks, PASS=all(nearchecks.values()), eigenvalues=near['eigenvalues'], eigenpair_relative_residuals=near['eigenpair_relative_residuals'], orthogonality_error=near['eigenvector_orthogonality_error'], cert_status=near['cert_status'], overall_status=near['overall_status'])
    reg_dump('adversarial_near_zero_record.json', near)
    np.savez_compressed(REGRESSION_OUT / 'adversarial_near_zero_arrays.npz', H_diagonal=diagonal, u_final=zero, eigenvalues=nearobj['eigenvalues'], eigenvectors=nearobj['eigenvectors'])
    truth = []
    for vals, expected in [([-2, -1, 1, 2, 3, 4], 'wrong_index'), ([1, 2, 3, 4, 5, 6], 'wrong_index'), ([-2, 1, 2, 3, 4, 5], 'verified_index1'), ([-2, 1e-6, 2, 3, 4, 5], 'verification_unresolved')]:
        got = classify_index_certificate(vals, np.zeros(6), np.zeros(6), 0.0, True)
        truth.append(got['cert_status'] == expected)
    truth.append(classify_index_certificate([-2, 1, 2, 3, 4, 5], np.zeros(6), np.zeros(6), 0, False)['cert_status'] == 'verification_unresolved')
    truth.append(classify_index_certificate([-2, 1], np.zeros(2), np.zeros(2), 0, True)['cert_status'] == 'verification_unresolved')
    truth.append(classify_index_certificate([-2, 1, 2, 3, 4, 5], np.ones(6), np.ones(6), 0, True)['cert_status'] == 'verification_unresolved')
    extra = dict(name='certification_classification_truth_table', PASS=all(truth), checks={'all_resolved_and_unresolved_cases': all(truth)})
    results = [badresult, nearresult, extra]
    reg_dump('adversarial_certification_results.json', results)
    return results

def reg_case_regression(baseline, N, p, method):
    cfg = configuration(N, p, method=method, purpose='regression')
    case = f'{method}_N{N}_p{p}'
    A = baseline['build_laplacian_2d'](N)
    u, h = baseline['make_u0'](N, 1.2)
    init = reg_baseline_start(baseline, cfg)
    oldcfg = baseline['B_CFG'].copy()
    if method == 'pHiSD_H1':
        ref = baseline['run_h1_fixed'](u, A, p, h, 1.0, oldcfg, False, False)
    else:
        ref = baseline['run_standard_fixed'](u, A, p, h, 0.0001, 0.0001, baseline['A_STD_CFG'].copy(), 60000)
    reg_dump(case + '_baseline_snapshot_record.json', {k: v for k, v in ref.items() if not isinstance(v, np.ndarray)})
    np.savez_compressed(REGRESSION_OUT / (case + '_baseline_snapshot_arrays.npz'), **{k: v for k, v in ref.items() if isinstance(v, np.ndarray)})
    rec, obj = solve(cfg)
    reg_save_solver_snapshot(case + '_solver_snapshot', rec, obj)
    oldhist = ref['grad_norm_history']
    newhist = obj['residual_history']
    equal_len = len(oldhist) == len(newhist)
    checks = {'both_success': ref['status'] == 'success' and rec['solver_status'] == 'success', 'baseline_initial_eigensolve_completed': not ref.get('note'), 'equal_updates': ref['iter'] - 1 == rec['counters']['n_outer_updates'], 'history_close': equal_len and np.allclose(oldhist, newhist, rtol=REGRESSION_TOLERANCES['history_rtol'], atol=REGRESSION_TOLERANCES['history_atol']), 'state_close': reg_relative(obj['u_final'], ref['u_final']) <= REGRESSION_TOLERANCES['state_relative']}
    max_abs = float(np.max(np.abs(oldhist - newhist))) if equal_len else None
    max_scaled = float(np.max(np.abs(oldhist - newhist) / (REGRESSION_TOLERANCES['history_atol'] + REGRESSION_TOLERANCES['history_rtol'] * np.abs(oldhist)))) if equal_len else None
    metrics = {'baseline_status': ref['status'], 'new_status': rec['solver_status'], 'baseline_legacy_iter': ref['iter'], 'new_history_length': len(newhist), 'baseline_actual_updates': ref['iter'] - 1, 'new_actual_updates': rec['counters']['n_outer_updates'], 'state_relative_error': reg_relative(obj['u_final'], ref['u_final']), 'history_max_absolute_error': max_abs, 'history_max_tolerance_fraction': max_scaled, 'baseline_final_residual': float(oldhist[-1]), 'new_final_residual': float(newhist[-1]), 'same_ARPACK_start_sha256': sha_bytes(init.tobytes()), 'new_recorded_start_sha256': rec['init_start_vector_sha256']}
    checks['frozen_source_unchanged'] = reg_source_unchanged()
    checks['same_ARPACK_start_hash'] = metrics['same_ARPACK_start_sha256'] == metrics['new_recorded_start_sha256']
    on_rec, on_obj = solve(configuration(N, p, method=method, profile=True), initial_frame=obj['v_initial'])
    reg_save_solver_snapshot(case + '_timer_on_snapshot', on_rec, on_obj)
    off_rec, off_obj = solve(configuration(N, p, method=method, profile=False), initial_frame=obj['v_initial'])
    reg_save_solver_snapshot(case + '_timer_off_snapshot', off_rec, off_obj)
    checks['timer_toggle_exact_state'] = np.array_equal(on_obj['u_final'], off_obj['u_final'])
    checks['timer_toggle_exact_history'] = np.array_equal(on_obj['residual_history'], off_obj['residual_history'])
    checks['timer_toggle_exact_frame'] = np.array_equal(on_obj['v_final'], off_obj['v_final'])
    checks['timer_toggle_identical_counters'] = on_rec['counters'] == off_rec['counters']
    checks['timer_toggle_identical_status'] = on_rec['solver_status'] == off_rec['solver_status']
    certify(rec, obj)
    checks['ordinary_index_certificate'] = rec['overall_status'] == 'verified_index1'
    checks['fresh_endpoint_residual_matches_history'] = rec['fresh_final_residual'] == rec['final_residual']
    structural_diagnostics(rec, obj)
    result = {'name': case, 'metrics': metrics, 'checks': checks, 'PASS': bool(all(checks.values()))}
    reg_dump(case + '_comparison.json', result)
    reg_dump(case + '_driver_record.json', rec)
    reg_dump(case + '_timer_on_record.json', on_rec)
    reg_dump(case + '_timer_off_record.json', off_rec)
    ref_record = {k: v for k, v in ref.items() if not isinstance(v, np.ndarray)}
    ref_record['explicit_ARPACK_start_sha256'] = sha_bytes(init.tobytes())
    reg_dump(case + '_baseline_record.json', ref_record)
    np.savez_compressed(REGRESSION_OUT / (case + '_arrays.npz'), baseline_u_final=ref['u_final'], new_u_final=obj['u_final'], baseline_residual_history=oldhist, new_residual_history=newhist, new_v_initial=obj['v_initial'], new_v_final=obj['v_final'], eigenvalues=obj['eigenvalues'], eigenvectors=obj['eigenvectors'], timer_on_u=on_obj['u_final'], timer_off_u=off_obj['u_final'], timer_on_history=on_obj['residual_history'], timer_off_history=off_obj['residual_history'])
    print(json.dumps({'case': case, 'PASS': result['PASS'], 'metrics': metrics}), flush=True)
    return result

def run_regression():
    global REGRESSION_OUT, REGRESSION_SOURCE_HASH
    REGRESSION_OUT = BASE / 'regression'
    REGRESSION_OUT.mkdir(parents=True, exist_ok=False)
    REGRESSION_SOURCE_HASH = code_hash()
    reg_dump('frozen_tolerances.json', REGRESSION_TOLERANCES)
    reg_dump('embedded_baseline_provenance.json', {
        'single_file_source_sha256': REGRESSION_SOURCE_HASH,
        'baseline_mode': 'Independent embedded ordinary mathematical functions; no runtime source extraction, imports of project scripts, or historical data reads.',
        'initialization': 'Identical explicit PCG64 start derived from each current configuration; baseline sparse generalized eigsh independently factors M as in the original implementation.',
    })
    started = time.perf_counter()
    baseline = reg_load_baseline()
    results = []
    tasks = [('small_N8_tests', lambda: [reg_small_tests(baseline)]),
             ('certification_adversarial_tests', reg_certification_adversarial_tests)]
    tasks.extend((f'{method}_N{N}_p{p}', lambda N=N, p=p, method=method: [reg_case_regression(baseline, N, p, method)])
                 for method, N, p in [('pHiSD_H1', N, p) for N in (64, 128) for p in (3, 5)]
                                      + [('standard_HiSD', 128, p) for p in (3, 5)])
    for name, execute in tasks:
        print(json.dumps({'regression_case_started': name}), flush=True)
        try:
            generated = execute()
            results.extend(generated)
        except Exception as ex:
            failure = dict(name=name, PASS=False, checks={'completed_without_exception': False},
                           exception=dict(type=type(ex).__name__, message=str(ex), traceback=traceback.format_exc()))
            reg_dump(name + '_exception.json', failure)
            results.append(failure)
        print(json.dumps(clean_json({'regression_case_completed': name, 'PASS': results[-1]['PASS']})), flush=True)
        if not reg_source_unchanged():
            results.append(dict(name='frozen_source_guard', PASS=False,
                                checks={'source_hash_unchanged': False}))
            break
    consistent = reg_source_unchanged()
    report = dict(gate=1, PASS=bool(all(item['PASS'] for item in results) and consistent),
                  code_hash=REGRESSION_SOURCE_HASH, source_hash_at_completion=code_hash(),
                  source_hash_consistent=consistent, tolerances=REGRESSION_TOLERANCES,
                  baseline_mode='Independent embedded original mathematical functions with explicit reproducible eigsh starts.',
                  old_results_read=False, fresh_test_data_generated=True,
                  results=results, wall_seconds=time.perf_counter()-started,
                  raw_directory=str(REGRESSION_OUT),
                  limitations=['Regression runs reuse the current process and are excluded from formal timing statistics.',
                               'Dense eigendecomposition is limited to the N=8 unit tests.',
                               'Timer equivalence pairs share an explicit initial frame; formal workers independently compute their own frames.'])
    write_json(BASE / 'REGRESSION_REPORT.json', report)
    print(json.dumps({'REGRESSION_PASS': report['PASS'], 'wall_seconds': report['wall_seconds']}), flush=True)
    return report


def run_validator_tests(record):
    untouched_input = copy.deepcopy(record)
    r = copy.deepcopy(normalize_record(record))
    checks, observations = {}, {}
    checks['fresh_N16_smoke_fixture'] = (value_of(r, 'N') == 16 and value_of(r, 'purpose') == 'smoke')
    if not checks['fresh_N16_smoke_fixture']:
        report = dict(PASS=False, checks=checks, failure='Current-run N=16 smoke fixture is required.', code_hash=code_hash())
        write_json(BASE / 'VALIDATOR_TEST_REPORT.json', report)
        return report
    fixture_errors = validate_record(r)
    observations['unmodified_fixture_errors'] = fixture_errors
    checks['saved_state_eigenpairs_and_hash_valid'] = not fixture_errors
    expected_warning = smoke_sampling_resolution_limit(r)
    warnings = record_warnings(r)
    checks['short_smoke_sampling_limit_explicit'] = (bool(warnings) == expected_warning)
    observations['fresh_fixture_warnings'] = warnings

    def rejects(name, mutate, fragment, arrays=False):
        q = copy.deepcopy(r)
        mutate(q)
        try:
            errors = validate_record(q, arrays)
            checks[name] = any(fragment in error for error in errors)
            observations[name] = {'expected_fragment': fragment, 'errors': errors}
        except Exception as ex:
            checks[name] = False
            observations[name] = {'unexpected_exception': type(ex).__name__ + ': ' + str(ex)}

    rejects('frozen_parameter_mutation_detected', lambda q: q['config']['protocol'].update(alpha=2.0), 'frozen protocol')
    rejects('actual_state_counter_mutation_detected', lambda q: q['counters'].update(n_outer_updates=q['counters']['n_outer_updates'] + 7), 'actual state updates')
    rejects('double_counted_eig_cost_detected', lambda q: q['timings'].update(T_total=q['timings']['T_total'] + q['timings']['T_eig_inclusive']), 'identity failed')
    rejects('false_eigenpair_acceptance_detected', lambda q: q['eigenpair_relative_residuals'].__setitem__(0, 1.0), 'residual acceptance')
    rejects('falsified_sparse_fill_detected', lambda q: q['structure'].update(fill_ratio=1.0), 'LU fill ratio')
    rejects('falsified_RSS_peak_detected', lambda q: q.update(solver_peak_rss_bytes=123), 'sampled peak disagrees')

    def missing_dry_samples(q):
        q['config']['purpose'] = 'dry_run'
        if 'purpose' in q:
            q['purpose'] = 'dry_run'
        q['monitor']['numerical_sample_count'] = 0
        q['solver_peak_rss_bytes'] = None
        q['full_peak_rss_bytes'] = None
    rejects('dry_run_cannot_use_short_smoke_exception', missing_dry_samples, 'sampled native RSS')
    rejects('array_file_hash_mutation_detected', lambda q: q.update(arrays_sha256='0' * 64), 'SHA-256', arrays=True)

    refused = {k: copy.deepcopy(r[k]) for k in ('N', 'n', 'p', 'method', 'config')}
    refused['config']['purpose'] = 'production'
    refused.update(solver_status='not_run_resource_limit', cert_status='not_run', overall_status='not_run_resource_limit')
    refused['config']['config_hash'] = canonical_config_hash(refused['config'])
    refusal_errors = validate_record(refused)
    checks['resource_refusal_is_not_data_corruption'] = not refusal_errors
    observations['resource_refusal_errors'] = refusal_errors

    def labeled(source, label, purpose, profile):
        q = copy.deepcopy(source)
        q['config'].update(purpose=purpose, run_id=label, profile=profile)
        q.update(run_id=label, purpose=purpose, profile=profile)
        q['config']['config_hash'] = canonical_config_hash(q['config'])
        if 'config_hash' in q:
            q['config_hash'] = q['config']['config_hash']
        return q
    profile = labeled(r, 'validator_profile', 'production', True)
    clean = labeled(r, 'validator_clean', 'production', False)
    pilot = labeled(r, 'validator_pilot', 'pilot', True)
    summaries = summarize([flatten(x) for x in (profile, clean, pilot)])
    checks['production_only_and_timer_modes_separate'] = len(summaries) == 2 and all(s['record_count'] == 1 for s in summaries)
    refused = labeled(refused, 'validator_refused', 'production', True)
    summaries = summarize([flatten(profile), flatten(refused)])
    checks['failed_denominator_retained'] = (len(summaries) == 1 and summaries[0]['record_count'] == 2
                                            and summaries[0]['verified_repeat_count'] == 1
                                            and summaries[0]['failed_or_censored_count'] == 1)
    planned = {'cases': [profile['config'], clean['config'], refused['config']]}
    matrix = production_check([profile, clean], planned)
    checks['missing_planned_record_visible'] = (matrix['missing_run_ids'] == ['validator_refused']
                                                and not matrix['formal_matrix_complete'])
    observations['synthetic_missing_record_matrix'] = matrix
    checks['fixture_immutable'] = clean_json(record) == clean_json(untouched_input)
    report = dict(PASS=bool(all(checks.values())), fixture_run_id=value_of(r, 'run_id'),
                  fixture_arrays_path=r.get('arrays_path'), code_hash=code_hash(),
                  fixture_origin='Fresh N=16 smoke worker from this execution', checks=checks,
                  observations=observations, numerical_runs_performed=0,
                  description='Mutations are in-memory copies. Endpoint/eigenpair hash checks and raw RSS reads use only current-run evidence; no historical fixture is read.')
    write_json(BASE / 'VALIDATOR_TEST_REPORT.json', report)
    print(json.dumps({'VALIDATOR_TEST_PASS': report['PASS']}), flush=True)
    return report


def gate_save(name, value):
    path = (BASE / name).resolve()
    if BASE.resolve() not in path.parents:
        raise ValueError('Gate output must remain inside the current result directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(clean_json(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return value


def gate_read(name, require_current_source=True):
    path = (BASE / name).resolve()
    if BASE.resolve() not in path.parents:
        raise ValueError('Gate input must belong to this invocation')
    result = json.loads(path.read_text(encoding='utf-8'))
    if require_current_source and result.get('code_hash') != code_hash():
        raise RuntimeError('Gate evidence source differs from current single-file source: ' + name)
    return result


def gate_log(message):
    line = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + message
    print(line, flush=True)


def gate_fingerprint(config):
    result = dict(config)
    result.pop('config_hash', None)
    result['config_hash'] = hashlib.sha256(json.dumps(result, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return result


def gate_configuration(N, p, purpose, method='pHiSD_H1', profile=True, rep=1, **extra):
    if N > max(MANUSCRIPT_GRIDS):
        raise ValueError('This single-file experiment is bounded at N=256')
    mode = 'profile' if profile else 'clean'
    ident = f'{purpose}_{method}_N{N}_p{p}_{mode}_r{rep}_a1'
    config = configuration(N, p, method, profile, run_id=ident, purpose=purpose,
        repetition=rep, attempt_id=1, timing_mode=mode,
        enforce_contention=(purpose not in ('smoke', 'dry_run')),
        expected_code_hash=code_hash(), code_sha256=code_hash(), **extra)
    return gate_fingerprint(config)


def gate_remaining_wall_s():
    budget = gate_read('resource_budget.json', require_current_source=False)
    if float(budget['active_wall_budget_s']) > 10800:
        raise RuntimeError('Resource budget exceeds the authorized three-hour maximum')
    return float(budget['start_unix']) + float(budget['active_wall_budget_s']) - time.time()


def gate_validate(record, require_verified=True):
    errors = validate_record(record)
    if isinstance(errors, dict):
        valid = errors.get('pass', errors.get('valid', not errors.get('errors')))
    else:
        valid = not errors
    if not valid:
        raise RuntimeError('Gate record validation failed: ' + str(errors))
    if require_verified and record.get('overall_status') != 'verified_index1':
        reason = _run_failure_details(record).get('reason')
        raise RuntimeError('Gate numerical outcome is ' + str(record.get('overall_status'))
                           + (': ' + str(reason) if reason else ''))
    return errors


def gate_execute_config(config):
    ident = config['run_id']
    if config.get('expected_code_hash') != code_hash():
        raise RuntimeError('Source changed after configuration freeze: ' + ident)
    if config != gate_fingerprint(config):
        raise RuntimeError('Configuration fingerprint mismatch: ' + ident)
    if (BASE / 'runs' / (ident + '.monitor.json')).exists():
        raise FileExistsError('A fresh invocation never resumes existing evidence: ' + ident)
    gate_log('START ' + ident)
    record = run_case(config)
    if record.get('overall_status') == 'verified_index1':
        if record.get('code_hash') != config['expected_code_hash']:
            raise RuntimeError('Worker source differs from the frozen source: ' + ident)
    if code_hash() != config['expected_code_hash']:
        raise RuntimeError('Single-file source changed while this case ran: ' + ident)
    timings = record.get('timings', {})
    gate_log('END ' + ident + ' ' + str(record.get('overall_status')) +
        ' total=' + str(timings.get('T_total')) + ' cert=' + str(timings.get('T_cert')) +
        ' rss=' + str(record.get('full_peak_rss_bytes')))
    return record


def gate_case(N, p, purpose, method='pHiSD_H1', profile=True, rep=1, **extra):
    return gate_execute_config(gate_configuration(N, p, purpose, method, profile, rep, **extra))


def gate_record_not_run(config, reason, decision=None):
    record = dict(run_id=config['run_id'], run_config_id=config['run_config_id'],
        config=config, code_hash=code_hash(), N=config['N'], n=config['N'] ** 2,
        p=config['p'], method=config['method'], profile=config['profile'],
        solver_status='not_run_resource_limit', cert_status='not_run',
        overall_status='not_run_resource_limit', resource_decision=decision,
        monitor=dict(stop_reason=reason, worker_pid=None, worker_start_method=None,
            record_path=str(BASE / 'runs' / (config['run_id'] + '.monitor.json')),
            expected_code_hash=config['expected_code_hash']),
        not_run_reason=reason, observations=None)
    gate_save('runs/' + config['run_id'] + '.monitor.json', record)
    gate_log('NOT RUN ' + config['run_id'] + ': ' + reason)
    return record


def gate_current_records(purposes=None):
    records = []
    for path in sorted((BASE / 'runs').glob('*.monitor.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if purposes is not None and record.get('config', {}).get('purpose') not in purposes:
            continue
        if record.get('overall_status') != 'verified_index1':
            continue
        if record.get('code_hash') != code_hash():
            raise RuntimeError('Current result directory contains a different source series')
        if not record.get('timings'):
            raise RuntimeError('Verified current record is missing timing evidence')
        records.append(record)
    return records


def gate_dry():
    regression = gate_read('REGRESSION_REPORT.json', require_current_source=False)
    if regression.get('PASS') is not True:
        raise RuntimeError('Regression has not passed in the current result directory')
    runs = []
    smoke = gate_case(64, 3, 'smoke')
    gate_validate(smoke)
    runs.append(smoke['config']['run_id'])
    for p in (3, 5):
        record = gate_case(256, p, 'dry_run')
        gate_validate(record)
        runs.append(record['config']['run_id'])
    return gate_save('gate2_dry_validation.json', dict(status='PASS', runs=runs,
        code_hash=code_hash(), evidence_scope='current invocation only'))


def gate_overhead(allow_one_recheck=True):
    gate_read('gate2_dry_validation.json')
    groups = [('pHiSD_H1', 64, p) for p in (3, 5)]
    groups += [('pHiSD_H1', 256, p) for p in (3, 5)]
    groups += [('standard_HiSD', 128, p) for p in (3, 5)]
    rule = ('At least 2 of 3 paired overhead fractions >0.05 and paired median '
        '>0.05 in any audited configuration selects separate clean and profile '
        'production; never subtract observed overhead from measured times.')

    def audit_group(method, N, p, round_id):
        pairs = []
        for pair in (1, 2, 3):
            order = (True, False) if pair % 2 else (False, True)
            by_mode = {}
            for profile in order:
                purpose = 'overhead' if round_id == 1 else 'overhead_check'
                record = gate_case(N, p, purpose, method, profile, pair)
                gate_validate(record)
                by_mode[profile] = record
            profile_total = by_mode[True]['timings']['T_total']
            clean_total = by_mode[False]['timings']['T_total']
            if clean_total <= 0:
                raise RuntimeError('Nonpositive clean timing cannot define overhead')
            pairs.append(dict(pair=pair, order=['profile' if q else 'clean' for q in order],
                profile_run=by_mode[True]['config']['run_id'],
                clean_run=by_mode[False]['config']['run_id'],
                profile_total=profile_total, clean_total=clean_total,
                overhead_fraction=profile_total / clean_total - 1))
        fractions = [pair['overhead_fraction'] for pair in pairs]
        median = statistics.median(fractions)
        return dict(method=method, N=N, p=p, round=round_id, pairs=pairs,
            median_overhead_fraction=median,
            sustained_over_5pct=sum(value > .05 for value in fractions) >= 2 and median > .05)

    first = [audit_group(*group, 1) for group in groups]
    gate_save('overhead_report.json', dict(status='PASS', round=1, groups=first,
        all_pairs=3 * len(first), decision_rule=rule, code_hash=code_hash(),
        use_clean_production=any(group['sustained_over_5pct'] for group in first)))
    selected = [
        group
        for group in first
        if allow_one_recheck and group['method'] == 'pHiSD_H1'
    ]
    second = [audit_group(group['method'], group['N'], group['p'], 2) for group in selected]
    if second:
        gate_save('overhead_check_report.json', dict(status='PASS', round=2,
            groups=second, all_pairs=3 * len(second), decision_rule=rule,
            reason='Fixed one-time recheck of all four p-HiSD groups, independent of first-round overhead values.',
            code_hash=code_hash()))
    replacement = {(group['method'], group['N'], group['p']): group for group in second}
    final = [replacement.get((group['method'], group['N'], group['p']), group) for group in first]
    report = dict(status='PASS', groups=final, decision_rule=rule, code_hash=code_hash(),
        use_clean_production=any(group['sustained_over_5pct'] for group in final),
        finite_recheck=dict(allowed=bool(allow_one_recheck), maximum_additional_rounds=1,
            trigger='When enabled, recheck all four p-HiSD groups once; standard HiSD uses its first-round results.',
            selected_groups=[dict(method=g['method'], N=g['N'], p=g['p']) for g in selected],
            final_group_rule='Use the complete second three-pair round when performed; retain both rounds.'),
        history_timings_excluded=True)
    gate_save('overhead_decision.json', report)
    gate_log('Overhead decision use_clean_production=' + str(report['use_clean_production']))
    return report


def gate_phase_prediction(records, method, N):
    matching = [r for r in records if r['method'] == method]
    if not matching:
        raise RuntimeError('Missing current measured resource reference for ' + method)
    above = sorted({r['N'] for r in matching if r['N'] >= N})
    source_N = above[0] if above else max(r['N'] for r in matching)
    source = [r for r in matching if r['N'] == source_N]
    ratio = (N / source_N) ** 2
    solver = max(r['timings']['T_total'] for r in source) * max(1, ratio ** 1.5) * 1.5
    cert = max(r['timings']['T_cert'] for r in source) * max(1, ratio ** 1.6) * 1.5
    return dict(source_N=source_N, source_runs=[r['config']['run_id'] for r in source],
        ratio_n=ratio, predicted_solver_s=solver, predicted_cert_s=cert,
        predicted_worker_wall_s=solver + cert + 3, time_safety_margin=1.5)


def gate_freeze():
    overhead = gate_read('overhead_decision.json')
    clean = overhead['use_clean_production']
    modes = [False, True] if clean else [True]
    grids = list(MANUSCRIPT_GRIDS)
    cases = []
    for method, sizes in [('pHiSD_H1', grids), ('standard_HiSD', [128])]:
        for N in sizes:
            for p in (3, 5):
                for profile in modes:
                    for rep in (1, 2, 3):
                        cases.append(gate_configuration(N, p, 'production', method, profile, rep))
    result = dict(frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        code_hash=code_hash(), protocol=PROTOCOL, grids=grids,
        p_values=[3, 5], repetitions=3, timing_mode='clean_and_profile' if clean else 'profile',
        cases=cases, requested_case_count=len(cases),
        resource_limits=dict(max_N=max(MANUSCRIPT_GRIDS), active_wall_budget_s=10800, solver_s=900, cert_s=900,
            rss='min(55% physical RAM,75% pre-run available RAM)'),
        seed_rule='SHA256(method_N_p:phase) first 4 bytes little endian; PCG64 standard_normal float64; independent init/cert; same seed per timing repetition',
        certification=dict(type='full-space ordinary eigsh', k=6, which='SA', sigma=None,
            tol=1e-8, maxiter=3000, ncv=20, relative_residual=1e-7, orthogonality=1e-8, delta_ind=1e-6),
        history_timings_excluded=True,
        initialization_retry=False, fresh_worker_start_method='spawn',
        mode_grouping='Separate method,N,p,timing_mode groups; exactly 3 fresh repetitions requested per group')
    gate_save('production_config.json', result)
    gate_log('Frozen production: ' + str(len(cases)) + ' requested cases; grids=' + str(grids))
    return result


def gate_production():
    frozen = gate_read('production_config.json')
    measured = gate_current_records({'smoke', 'dry_run', 'overhead', 'overhead_check'})
    required = sum(gate_phase_prediction(measured, c['method'], c['N'])['predicted_worker_wall_s'] for c in frozen['cases'])
    remaining = gate_remaining_wall_s()
    preflight = dict(required_remaining_matrix_s=required, remaining_authorized_wall_s=remaining,
        safe=required < remaining, code_hash=code_hash(), resources=resources_snapshot(BASE))
    gate_save('production_preflight.json', preflight)
    outcomes = []
    blocked = None if preflight['safe'] else 'complete production matrix exceeds current remaining wall budget'
    for config in frozen['cases']:
        if frozen['code_hash'] != code_hash():
            raise RuntimeError('Source changed after production freeze; stop this timing series')
        if blocked:
            record = gate_record_not_run(config, blocked, preflight)
        else:
            record = gate_execute_config(config)
            if record.get('overall_status') == 'verified_index1':
                gate_validate(record)
            elif record.get('overall_status') in ('not_run_resource_limit', 'memory_limit',
                    'timeout_solver', 'timeout_cert', 'timeout_startup', 'timeout_postprocessing'):
                blocked = 'no dependent timing launched after current resource outcome: ' + str(record.get('overall_status'))
            else:
                blocked = 'no dependent timing launched after nonverified current numerical outcome: ' + str(record.get('overall_status'))
        outcomes.append(dict(run_id=config['run_id'], method=config['method'], N=config['N'],
            p=config['p'], timing_mode=config['timing_mode'], repetition=config['repetition'],
            overall_status=record.get('overall_status')))
    successful = sum(r['overall_status'] == 'verified_index1' for r in outcomes)
    result = dict(code_hash=code_hash(), requested_case_count=len(frozen['cases']),
        verified_case_count=successful, complete=successful == len(frozen['cases']), outcomes=outcomes,
        final_blocking_reason=blocked, all_failures_and_refusals_retained=True)
    gate_save('production_execution_report.json', result)
    gate_log('Production finished: ' + str(successful) + '/' + str(len(frozen['cases'])) + ' verified requested runs')
    return result


EXPECTED = dict(alpha=1.0, amp=1.2, eta_phisd=0.5, eta_standard=1e-4,
    tau=1e-4, J=5, tol=1e-6, max_phisd=2000000, max_standard=2000000,
    div_grad=1e8, div_E=1e12, eig_tol=1e-8, eig_maxiter=3000,
    eig_ncv=20, cert_k=6, cert_resid=1e-7, cert_orth=1e-8, delta_ind=1e-6)
CENSORED = {'timeout_solver', 'timeout_cert', 'memory_limit', 'not_run_resource_limit', 'not_run_contention', 'worker_failed', 'worker_crashed', 'budget_exhausted'}
TIMERS = ['T_total', 'T_pc_setup_update', 'T_pc_setup_initial', 'T_M_build', 'T_factor',
    'T_pc_update', 'T_pc_apply_solve', 'T_M_apply', 'T_M_solve', 'T_eig_inclusive',
    'T_eig_init_inclusive', 'T_frame_updates_inclusive', 'T_pc_inside_eig',
    'T_eig_exclusive_pc', 'T_other', 'T_orth_inclusive', 'T_cert', 'T_cert_eig',
    'T_total_with_cert', 'T_problem_setup', 'T_outer_inclusive', 'T_driver_other']
COUNTERS = ['n_outer_updates', 'n_residual_evals', 'n_grad_evals', 'n_energy_evals',
    'n_H_assemblies', 'n_HVP', 'n_A_matvec', 'n_M_apply_calls', 'n_M_solve_calls',
    'n_M_solve_rhs', 'n_frame_inner_steps', 'n_orth_calls', 'n_factorizations', 'n_pc_updates']
MEMORY = ['baseline_rss_bytes', 'solver_peak_rss_bytes', 'full_peak_rss_bytes',
    'solver_start_highwater_bytes', 'solver_end_highwater_bytes', 'cert_end_highwater_bytes']
STRUCTURE = ['nnz_A', 'nnz_M', 'nnz_H_final', 'nnz_L', 'nnz_U', 'F_n', 'fill_ratio',
    'sparse_A_payload_bytes', 'sparse_M_payload_bytes', 'factor_exported_array_payload_bytes']


def safe_path(value):
    p = Path(value)
    if not p.is_absolute(): p = BASE / p
    p = p.resolve()
    if not p.is_relative_to(Path(BASE).resolve()):
        raise ValueError('Path outside current run directory: ' + str(p))
    return p


def sha_file(path):
    h = hashlib.sha256()
    with safe_path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def close(a, b, atol=1e-8, rtol=1e-10):
    return finite(a) and finite(b) and abs(a-b) <= atol + rtol*max(abs(a), abs(b))


def canonical_config_hash(cfg):
    body=dict(cfg)
    body.pop('config_hash',None)
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def config_of(r): return r.get('config', {})

def value_of(r, key, default=None): return r.get(key, config_of(r).get(key, default))

def mode_of(r): return 'profile' if config_of(r).get('profile', r.get('profile', True)) else 'clean'

def normalize_record(raw):
    r = dict(raw.get('record', {}))
    r.update({k:v for k,v in raw.items() if k != 'record'})
    return r


def smoke_sampling_resolution_limit(r):
    t=r.get('timings',{});mon=r.get('monitor',{})
    missing_solver = r.get('solver_peak_rss_bytes') is None
    missing_all = r.get('full_peak_rss_bytes') is None
    short_solver = finite(t.get('T_total')) and 0<t['T_total']<SAMPLE_INTERVAL_S
    short_all = (mon.get('numerical_sample_count')==0 and finite(t.get('T_total_with_cert'))
                 and 0<t['T_total_with_cert']<SAMPLE_INTERVAL_S)
    return (value_of(r,'purpose')=='smoke' and (missing_solver or missing_all)
            and (not missing_solver or short_solver) and (not missing_all or short_all)
            and finite(r.get('solver_end_highwater_bytes')) and r['solver_end_highwater_bytes']>0
            and finite(r.get('cert_end_highwater_bytes')) and r['cert_end_highwater_bytes']>0)


def record_warnings(r):
    warnings=[]
    if smoke_sampling_resolution_limit(r):
        warnings.append('A smoke numerical phase shorter than the requested 20 ms RSS sampling period has no captured RSS sample. Null sampled peaks are retained; cumulative OS high-water is available and is not substituted for a sampled phase peak.')
    return warnings


def validate_record(r, check_arrays=True):
    errors = []
    def check(condition, message):
        if not condition: errors.append(message)
    def eq(a,b,message): check(close(a,b), message + f' ({a!r} vs {b!r})')
    r = normalize_record(r)
    cfg=config_of(r); status=r.get('overall_status'); solver=r.get('solver_status'); cert=r.get('cert_status')
    for identity_key in ('N','p','method','run_id','run_config_id','profile'):
        if identity_key in r and identity_key in cfg:check(r[identity_key]==cfg[identity_key],'record/config identity mismatch '+identity_key)
    expected_config_hash=canonical_config_hash(cfg)
    for location,recorded_hash in (('config',cfg.get('config_hash')),('record',r.get('config_hash'))):
        if recorded_hash is not None:check(recorded_hash==expected_config_hash,location+' canonical config SHA-256 mismatch')
    if value_of(r,'purpose')=='production':
        check(cfg.get('config_hash') is not None or r.get('config_hash') is not None,'formal record missing recorded config_hash')
    N=value_of(r,'N'); p=value_of(r,'p'); method=value_of(r,'method')
    check(isinstance(N,int) and 1 <= N <= 1024, 'N outside supported protocol')
    check(p in (3,5), 'p outside frozen set')
    check(method in ('pHiSD_H1','standard_HiSD'), 'unsupported method')
    if not isinstance(N,int) or N < 1: return errors
    check(r.get('n',N*N)==N*N, 'n != N^2')
    if cfg.get('protocol') is not None: check(cfg['protocol']==EXPECTED, 'frozen protocol mismatch')
    elif status not in CENSORED: check(False,'missing frozen protocol')
    if status and status.startswith('not_run'): return errors
    finished = solver in ('success','max_iter','numerical_divergence') and bool(r.get('timings'))
    completed = finished and status not in CENSORED
    t=r.get('timings',{}); c=r.get('counters',{}); phases=r.get('phase_counters',{})
    if t:
        check(r.get('includes_operator_assembly') is True,'T_total assembly flag missing/false')
        check(r.get('includes_certification') is False,'T_total certification flag missing/true')
        for k,v in t.items():
            if v is not None: check(finite(v) and v >= -1e-8, 'nonfinite or negative timer '+k)
        check(finite(t.get('T_total')) and t['T_total']>0,'missing/nonpositive T_total')
        if all(finite(t.get(k)) for k in ('T_total','T_cert','T_total_with_cert')):
            eq(t['T_total_with_cert'], t['T_total']+t['T_cert'], 'solver+cert total mismatch')
        if finite(t.get('T_cert_eig')) and finite(t.get('T_cert')):
            check(t['T_cert_eig']<=t['T_cert']+1e-8,'cert eig time outside cert parent')
        for total,parts in [('T_pc_setup_update',['T_pc_setup_initial','T_pc_update']),
                            ('T_eig_inclusive',['T_eig_init_inclusive','T_frame_updates_inclusive']),
                            ('T_total',['T_problem_setup','T_pc_setup_initial','T_eig_init_inclusive','T_outer_inclusive','T_driver_other'])]:
            if all(finite(t.get(k)) for k in [total]+parts): eq(t[total],sum(t[k] for k in parts),total+' stage identity failed')
        if finite(t.get('T_M_build')) and finite(t.get('T_factor')) and finite(t.get('T_pc_setup_initial')):
            check(t['T_M_build']+t['T_factor']<=t['T_pc_setup_initial']+1e-8,'pc children exceed setup parent')
        if mode_of(r)=='profile':
            for total,parts in [('T_pc_apply_solve',['T_M_apply','T_M_solve']),
                               ('T_total',['T_pc_setup_update','T_pc_apply_solve','T_eig_exclusive_pc','T_other'])]:
                if all(finite(t.get(k)) for k in [total]+parts): eq(t[total],sum(t[k] for k in parts),total+' exclusive identity failed')
                elif completed: check(False,'profile is missing additive timer '+total)
            if all(finite(t.get(k)) for k in ('T_eig_exclusive_pc','T_eig_inclusive','T_pc_inside_eig')):
                eq(t['T_eig_exclusive_pc'],t['T_eig_inclusive']-t['T_pc_inside_eig'],'eig exclusive subtraction failed')
            ns=r.get('phase_timing_ns',{})
            if ns and finite(t.get('T_pc_inside_eig')):
                inside=sum(ns.get(ph+':T_'+op,0) for ph in ('init_eig','outer_frame') for op in ('M_apply','M_solve'))/1e9
                eq(t['T_pc_inside_eig'],inside,'inside-eig pc phase accounting failed')
                for op in ('M_apply','M_solve'):
                    val=sum(v for k,v in ns.items() if k.endswith(':T_'+op) and k.split(':')[0] not in ('cert','audit'))/1e9
                    eq(t.get('T_'+op),val,'solver phase accounting '+op)
            if finite(t.get('T_orth_inclusive')) and finite(t.get('T_eig_inclusive')):
                check(t['T_orth_inclusive']<=t['T_eig_inclusive']+1e-8,'normalization diagnostic exceeds eig parent')
        eq(t.get('T_pc_update',0),0,'fixed M update time nonzero')
    if c:
        for k,v in c.items(): check(isinstance(v,int) and v>=0,'noninteger/negative counter '+k)
        check(c.get('n_pc_updates',0)==0,'fixed M update count nonzero')
    if finished and not r.get('solver_exception'):
        for k in COUNTERS: check(k in c,'missing counter '+k)
        I=c.get('n_outer_updates',-1)
        check(r.get('history_length')==I+1,'history length != actual state updates + 1')
        check(r.get('legacy_iter')==r.get('history_length'),'legacy iteration mapping mismatch')
        check(c.get('n_residual_evals')==I+1,'residual evaluations != actual state updates + 1')
        check(c.get('n_grad_evals')==I+1,'gradient count != residual count')
        check(c.get('n_energy_evals')==I+1,'energy count != residual count')
        check(c.get('n_frame_inner_steps')==5*I,'inner frame count != J*I')
        f=phases.get('outer_frame',{}); s=phases.get('outer_state',{})
        check(f.get('n_HVP_rhs',f.get('n_HVP',0))==5*I,'actual outer HVP != J*I')
        check(f.get('n_orth_calls',0)==5*I,'frame normalization count != J*I')
        if method=='pHiSD_H1':
            check(c.get('n_factorizations')==1,'actual SuperLU factorization count != 1')
            check(phases.get('init_eig',{}).get('n_M_solve_calls',0)>0,'initial Minv was not called')
            check(f.get('n_M_solve_calls',0)+s.get('n_M_solve_calls',0)==6*I,'actual outer M-solves != (J+1)*I')
            check(f.get('n_M_apply_calls',0)+s.get('n_M_apply_calls',0)==10*I,'actual outer M-applications != 2*J*I')
        else:
            for k in ('n_factorizations','n_M_apply_calls','n_M_solve_calls','n_M_solve_rhs'):
                check(c.get(k,0)==0,'standard HiSD has nonzero '+k)
            for k in ('T_pc_setup_update','T_pc_setup_initial','T_M_build','T_factor','T_pc_update','T_pc_apply_solve','T_M_apply','T_M_solve'):
                eq(t.get(k,0),0,'standard HiSD has nonzero '+k)
        for k in COUNTERS:
            if k=='n_HVP':
                actual=sum(v.get('n_HVP_rhs',0) for ph,v in phases.items() if ph not in ('cert','audit'))
            else: actual=sum(v.get(k,0) for ph,v in phases.items() if ph not in ('cert','audit'))
            check(c.get(k)==actual,'aggregate/phase counter mismatch '+k)
    if solver=='success': check(finite(r.get('final_residual')) and r['final_residual']<1e-6,'solver success residual fails')
    if status=='verified_index1':
        check(solver=='success' and cert=='verified_index1','overall verified status inconsistent')
        for flag in ('endpoint_finite','endpoint_residual_pass','eigenpair_accuracy_pass','index_sign_resolved','eigensolver_converged'):
            check(r.get(flag) is True,'overall verified acceptance flag inconsistent: '+flag)
        check(finite(r.get('fresh_final_residual')) and r['fresh_final_residual']<1e-6,'verified endpoint residual fails')
    if cert=='verified_index1':
        check(r.get('eigensolver_converged') is True, 'verified certificate without normal eigensolver completion')
        vals=r.get('eigenvalues',[]); ab=r.get('eigenpair_absolute_residuals',[]); rel=r.get('eigenpair_relative_residuals',[])
        check(len(vals)==len(ab)==len(rel)==6,'certification does not contain six complete eigenpairs')
        if len(vals)==len(ab)==len(rel)==6:
            check(all(finite(x) for x in vals+ab+rel),'nonfinite cert values')
            if all(finite(x) for x in vals+ab+rel):
                check(vals==sorted(vals),'cert eigenvalues not algebraically sorted')
                check(all(0<=x<=1e-7 for x in rel),'eigenpair residual acceptance fails')
                check(vals[0]+10*ab[0]<-1e-6 and vals[1]-10*ab[1]>1e-6,'Morse index sign margin fails')
                for a,b,l in zip(ab,rel,vals): eq(b,a/max(1,abs(l)),'eigenpair relative residual normalization mismatch')
        check(finite(r.get('eigenvector_orthogonality_error')) and r['eigenvector_orthogonality_error']<=1e-8,'cert orthogonality acceptance fails')
        check(r.get('cert_sign_check') is True and r.get('cert_accuracy_check') is True,'cert boolean check inconsistent')
        check(r.get('numerical_index')==1,'numerical index label inconsistent')
        opts=r.get('options',{}).get('eigsh',{})
        check(opts.get('which')=='SA' and opts.get('sigma') is None and opts.get('ncv')==20 and opts.get('tol')==1e-8 and opts.get('maxiter')==3000,'frozen ordinary eigensolver options mismatch')
    st=r.get('structure',{})
    if st:
        expected_nnz=5*N*N-4*N
        if st.get('nnz_A') is not None: check(st['nnz_A']==expected_nnz,'A sparsity pattern mismatch')
        if method=='pHiSD_H1' and st.get('nnz_M') is not None:
            check(st['nnz_M']==expected_nnz,'M sparsity pattern mismatch')
            if all(finite(st.get(k)) for k in ('nnz_L','nnz_U','F_n')): eq(st['F_n'],st['nnz_L']+st['nnz_U'],'LU factor nnz sum mismatch')
            if finite(st.get('F_n')) and st.get('nnz_M',0)>0: eq(st.get('fill_ratio'),st['F_n']/st['nnz_M'],'LU fill ratio mismatch')
    if completed and r.get('arrays_path'):
        for k in ('baseline_rss_bytes','solver_peak_rss_bytes','full_peak_rss_bytes'):
            unresolved_smoke = k!='baseline_rss_bytes' and r.get(k) is None and smoke_sampling_resolution_limit(r)
            check(unresolved_smoke or (finite(r.get(k)) and r[k]>0),'missing/nonpositive sampled native RSS '+k)
        if finite(r.get('full_peak_rss_bytes')) and finite(r.get('solver_peak_rss_bytes')):
            check(r['full_peak_rss_bytes']>=r['solver_peak_rss_bytes'],'full sampled RSS lower than solver peak')
        if r.get('thread_environment'):
            check(all(r['thread_environment'].get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'non-single-thread environment')
    if completed and r.get('monitor'):
        errors.extend(validate_memory_samples(r))
    if check_arrays and r.get('arrays_path'):
        try: errors.extend(validate_arrays(r))
        except Exception as ex: errors.append('array validation exception: '+type(ex).__name__+': '+str(ex))
    elif check_arrays and status=='verified_index1': check(False,'verified record missing endpoint/eigenpair archive')
    return errors


def validate_memory_samples(r):
    errors=[];mon=r['monitor'];intervals=r.get('numerical_intervals_ns',{})
    if mon.get('worker_start_method')!='spawn':errors.append('worker was not an independent spawn process')
    for phase,key in (('solver','T_total'),('cert','T_cert')):
        if phase in intervals:
            bounds=intervals[phase]
            if len(bounds)!=2 or not all(isinstance(x,int) for x in bounds) or bounds[1]<bounds[0]:
                errors.append('invalid measured '+phase+' interval');continue
            if not close((bounds[1]-bounds[0])/1e9,r.get('timings',{}).get(key),1e-9,1e-12):errors.append(phase+' interval and timer differ')
    if not mon.get('rss_samples_path'):return errors+['missing raw native-RSS samples path']
    try:
        with safe_path(mon['rss_samples_path']).open(newline='', encoding='utf-8') as f:samples=list(csv.DictReader(f))
        groups={ph:[] for ph in ('solver','cert')}
        for row in samples:
            timestamp=int(row['perf_counter_ns']);rss=int(row['rss_bytes'])
            for ph in groups:
                bounds=intervals.get(ph)
                if bounds and bounds[0]<=timestamp<=bounds[1]:groups[ph].append(rss)
        pairs=[('solver_peak_rss_bytes',max(groups['solver'],default=None)),
               ('full_peak_rss_bytes',max(groups['solver']+groups['cert'],default=None))]
        for key,value in pairs:
            if r.get(key)!=value:errors.append('sampled peak disagrees with raw numerical-interval RSS: '+key)
    except Exception as ex:errors.append('raw RSS validation failed: '+str(ex))
    if value_of(r,'purpose')=='production' and mon.get('pre_run_contention',{}).get('blocked'):
        errors.append('production worker launched despite pre-run sustained contention')
    return errors


def validate_arrays(r):
    import numpy as np
    import scipy.sparse as sp
    err=[]
    def check(ok,msg):
        if not ok: err.append(msg)
    path=safe_path(r['arrays_path'])
    check(path.exists(),'saved array file missing')
    if not path.exists(): return err
    check(sha_file(path)==r.get('arrays_sha256'),'NPZ file SHA-256 mismatch')
    with np.load(path,allow_pickle=False) as z:
        N=r['N']; n=N*N; p=r['p']; h=np.pi/(N+1)
        if 'u_final' not in z: return err + ['saved endpoint missing']
        u=z['u_final'];check(u.shape==(n,),'endpoint shape mismatch');check(u.dtype==np.float64,'endpoint not float64')
        if u.shape!=(n,):return err
        if not np.isfinite(u).all():
            check(r.get('solver_status')!='success','success archive contains nonfinite endpoint')
            return err
        e=np.ones(N); T=sp.diags([-e,4*e,-e],[-1,0,1],shape=(N,N),format='csr')
        S=sp.diags([-e,-e],[-1,1],shape=(N,N),format='csr'); I=sp.eye(N,format='csr')
        A=((sp.kron(I,T,format='csr')+sp.kron(S,I,format='csr'))/h**2).tocsr()
        Au=A@u; res=float(h*np.linalg.norm(Au-u**p)); E=float(0.5*(u@Au)-np.sum(u**(p+1))/(p+1))
        # Independent sparse matvec is deterministic at fixed saved state. The
        # absolute 5e-12 tolerance permits floating point reduction roundoff.
        if finite(r.get('final_residual')): check(close(res,r['final_residual'],5e-12,1e-8),'saved state/residual mismatch')
        if finite(r.get('fresh_final_residual')): check(close(res,r['fresh_final_residual'],5e-12,1e-8),'saved state/fresh verification residual mismatch')
        if finite(r.get('energy')):check(close(E,r['energy'],1e-6,1e-11),'saved state/energy mismatch')
        hist=z['residual_history'];check(len(hist)==r.get('history_length'),'saved history length mismatch')
        if len(hist):check(close(float(hist[-1]),r.get('final_residual'),5e-12,1e-10),'saved final history residual mismatch')
        if 'energy_history' in z:check(len(z['energy_history'])==len(hist),'energy/residual history length mismatch')
        Mv=lambda v:A@v+v if r['method']=='pHiSD_H1' else v
        for key in ('v_final','v_initial'):
            if key in z:
                v=z[key];check(v.shape==(n,) and np.isfinite(v).all(),'invalid saved '+key)
                if v.shape==(n,) and np.isfinite(v).all():
                    normal=abs(float(v@Mv(v))-1)
                    check(normal<=1e-8,key+' normalization fails')
                    if key=='v_final' and finite(r.get('frame_normalization_error')):check(close(normal,r['frame_normalization_error'],1e-10,1e-5),'frame normalization diagnostic mismatch')
        cfg=config_of(r)
        for phase in ('init','cert'):
            recorded=r.get(phase+'_start_vector_sha256');seed=cfg.get('seed_'+phase)
            if recorded and seed is not None:
                vec=np.random.Generator(np.random.PCG64(seed)).standard_normal(n)
                check(hashlib.sha256(vec.tobytes()).hexdigest()==recorded,phase+' starting vector seed/hash mismatch')
        if r.get('cert_status')=='verified_index1':
            vals=z['eigenvalues'];Q=z['eigenvectors'];check(vals.shape==(6,) and Q.shape==(n,6),'saved cert eigenpair shape mismatch')
            if vals.shape==(6,) and Q.shape==(n,6):
                check(np.isfinite(vals).all() and np.isfinite(Q).all(),'saved cert eigenpairs nonfinite')
                H=A-sp.diags(p*u**(p-1),0,format='csr')
                absres=np.linalg.norm(H@Q-Q*vals,axis=0)
                rel=absres/np.maximum(1,np.abs(vals));orth=float(np.linalg.norm(Q.T@Q-np.eye(6),2))
                check(np.all(rel<=1e-7),'independent eigenpair residual acceptance fails')
                check(orth<=1e-8,'independent eigenvector orthogonality acceptance fails')
                check(np.max(np.abs(np.linalg.norm(Q,axis=0)-1))<=1e-8,'eigenvectors are not Euclidean unit vectors')
                check(bool(vals[0]+10*absres[0]<-1e-6 and vals[1]-10*absres[1]>1e-6),'independent index sign margin fails')
                check(np.allclose(vals,r['eigenvalues'],rtol=1e-12,atol=1e-12),'archive/JSON eigenvalues mismatch')
                # Assembly order can change cancellation at fine meshes, so
                # agreement tolerance is absolute and acceptance is rechecked.
                check(np.allclose(absres,r['eigenpair_absolute_residuals'],rtol=1e-3,atol=5e-9),'archive/JSON eigenpair residual diagnostics mismatch')
        return err


def read_json(path):
    return json.loads(safe_path(path).read_text(encoding='utf-8'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonstandard JSON constant '+x)))


def flatten(r,source=None):
    cfg=config_of(r)
    row={k:value_of(r,k) for k in ('run_id','attempt_id','run_config_id','purpose','N','n','p','method','code_hash')}
    row['n']=row['N']**2 if isinstance(row['N'],int) else row['n']
    row.update(timing_mode=mode_of(r),solver_status=r.get('solver_status'),cert_status=r.get('cert_status'),overall_status=r.get('overall_status'),
        final_residual=r.get('final_residual'),fresh_final_residual=r.get('fresh_final_residual'),history_length=r.get('history_length'),legacy_iter=r.get('legacy_iter'),
        arrays_path=r.get('arrays_path'),arrays_sha256=r.get('arrays_sha256'),raw_record_path=source)
    for k in TIMERS:row[k]=r.get('timings',{}).get(k)
    for k in COUNTERS:row[k]=r.get('counters',{}).get(k)
    for k in MEMORY:row[k]=r.get(k)
    for k in STRUCTURE:row[k]=r.get('structure',{}).get(k)
    for ph,co in r.get('phase_counters',{}).items():
        for k,v in co.items():row[ph+'__'+k]=v
    for k in ('reason','failure_reason','resource_reason','consumed_wall_s','wall_time_seconds','started_utc','python','numpy','scipy'):
        val=r.get(k)
        row[k]=json.dumps(val,ensure_ascii=False,sort_keys=True) if isinstance(val,(dict,list)) else val
    mon=r.get('monitor',{})
    row['consumed_wall_s']=r.get('consumed_wall_s',mon.get('worker_elapsed_wall_s'))
    row['failure_reason']=(r.get('failure_reason') or mon.get('stop_reason') or (r.get('resource_decision') or {}).get('reason')
                           or r.get('solver_exception',{}).get('message') or r.get('cert_exception',{}).get('message'))
    row['resource_decision']=json.dumps(r['resource_decision'],sort_keys=True) if r.get('resource_decision') else None
    row['worker_pid']=mon.get('worker_pid');row['worker_start_method']=mon.get('worker_start_method')
    row['monitor_code_sha256']=mon.get('monitor_code_sha256')
    row['config_hash']=r.get('config_hash') or cfg.get('config_hash')
    row['derived_config_hash']=canonical_config_hash(cfg)
    row['config_hash_provenance']='recorded in original run record/config' if row['config_hash'] is not None else 'not recorded; derived_config_hash computed during postprocessing only'
    row['derived_config_hash_provenance']='postprocessing SHA256 of config without config_hash using sorted compact JSON; not a run-time capture'
    row['cert_eigenvalues']=json.dumps(r.get('eigenvalues')) if r.get('eigenvalues') is not None else None
    row['cert_max_relative_eigenpair_residual']=max(r['eigenpair_relative_residuals']) if r.get('eigenpair_relative_residuals') else None
    row['cert_orthogonality_error']=r.get('eigenvector_orthogonality_error')
    return row


def write_csv(name,rows,fieldnames=None):
    if name in _FINAL_RESULT_FILES:
        rows = _publication_data(rows)
        if fieldnames is not None:
            fieldnames = list(_publication_data(dict.fromkeys(fieldnames)))
    if fieldnames is None:fieldnames=list(dict.fromkeys(k for row in rows for k in row))
    with (BASE/name).open('w',newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames or ['no_records']);w.writeheader();w.writerows(rows)


def statistics_for(rows,keys):
    out={}
    for k in keys:
        vals=[r[k] for r in rows if finite(r.get(k))]
        for label,value in [('median',statistics.median(vals) if vals else None),('min',min(vals) if vals else None),('max',max(vals) if vals else None)]:out[label+'_'+k]=value
        out['count_'+k]=len(vals)
    return out


# Report median/min/max over verified runs; retain failed attempts in outcome counts.
def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        if r.get('purpose')=='production':groups[(r['method'],r['p'],r['N'],r['timing_mode'])].append(r)
    summaries=[]
    metrics=TIMERS+COUNTERS+MEMORY+STRUCTURE+['final_residual','fresh_final_residual','history_length']
    for (method,p,N,mode),group in sorted(groups.items()):
        verified=[r for r in group if r['overall_status']=='verified_index1']
        solved=[r for r in group if r['solver_status']=='success']
        mixed_source = len({r.get('code_hash') for r in verified}) > 1
        if mixed_source:
            verified=[]; solved=[]
        row=dict(method=method,p=p,N=N,n=N*N,timing_mode=mode,record_count=len(group),verified_repeat_count=sum(r['overall_status']=='verified_index1' for r in group),solver_success_count=sum(r['solver_status']=='success' for r in group),
                 failed_or_censored_count=sum(r['overall_status']!='verified_index1' for r in group),status_counts=json.dumps(Counter(r.get('overall_status') for r in group),sort_keys=True),
                 timing_denominator='overall_status=verified_index1',run_ids=json.dumps([r['run_id'] for r in group]),code_hashes=json.dumps(sorted(set(r['code_hash'] for r in group if r.get('code_hash')))))
        row['mixed_source_statistics_withheld']=mixed_source
        if mixed_source: row['timing_denominator']='WITHHELD: mixed source versions; raw outcomes retained'
        row.update(statistics_for(verified,metrics))
        for k,v in statistics_for(solved,['T_total','T_cert','T_total_with_cert','n_outer_updates','final_residual']).items():row['solver_success_'+k]=v
        for k in MEMORY:
            vals=[r[k] for r in group if finite(r.get(k))];row['observed_max_all_outcomes_'+k]=max(vals) if vals else None
        summaries.append(row)
    return summaries


def _current_source_sha256():
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def production_check(records, prod):
    errors = []
    planned = [c.get('config', c) for c in prod.get('cases', [])]
    formal = [r for r in records if value_of(r, 'purpose') == 'production']
    by_id = defaultdict(list)
    for r in formal:
        by_id[value_of(r, 'run_id')].append(r)
    missing = []
    for cfg in planned:
        rid = cfg.get('run_id')
        if cfg.get('config_hash') != canonical_config_hash(cfg):
            errors.append('Frozen planned case missing/invalid canonical config hash: ' + str(rid))
        matches = by_id.get(rid, [])
        if not matches:
            missing.append(rid)
        if len(matches) > 1:
            errors.append('Duplicate formal run ID: ' + str(rid))
        for r in matches:
            if config_of(r) != cfg:
                errors.append('Production config differs from frozen case: ' + str(rid))
    planned_ids = {c.get('run_id') for c in planned}
    if len(planned_ids) != len(planned):
        errors.append('Duplicate run IDs in frozen production plan')
    if not planned:
        errors.append('Production matrix not frozen or cases list missing')
    for r in formal:
        if value_of(r, 'run_id') not in planned_ids:
            errors.append('Unplanned production record ' + str(value_of(r, 'run_id')))
    groups = defaultdict(list)
    for r in formal:
        groups[(value_of(r, 'method'), value_of(r, 'p'), value_of(r, 'N'), mode_of(r))].append(r)
    group_reports = []
    for (method, p, N, mode), members in sorted(groups.items()):
        launched = [r for r in members if r.get('monitor', {}).get('worker_pid') is not None]
        pids = [r['monitor']['worker_pid'] for r in launched]
        checks = dict(three_attempt_records=len(members) == 3 and len({value_of(r, 'run_id') for r in members}) == 3,
                      repetitions_1_2_3=sorted(config_of(r).get('repetition', -1) for r in members) == [1, 2, 3],
                      three_fresh_worker_pids=len(launched) == 3 and len(set(pids)) == 3,
                      all_spawn=all(r.get('monitor', {}).get('worker_start_method') == 'spawn' for r in launched),
                      all_workers_exit_zero=all(r.get('monitor', {}).get('worker_exitcode') == 0 for r in launched),
                      same_input_seeds=len({(config_of(r).get('seed_init'), config_of(r).get('seed_cert')) for r in members}) == 1,
                      init_cert_seeds_distinct=all(config_of(r).get('seed_init') != config_of(r).get('seed_cert') for r in members),
                      same_code_hash=len({r.get('code_hash') for r in launched}) <= 1,
                      same_init_vector=len({r.get('init_start_vector_sha256') for r in launched}) <= 1,
                      same_cert_vector=len({r.get('cert_start_vector_sha256') for r in launched}) <= 1)
        for name in ('same_input_seeds', 'init_cert_seeds_distinct', 'same_code_hash', 'same_init_vector', 'same_cert_vector'):
            if not checks[name]:
                errors.append('Production repeat integrity failure ' + str((method, p, N, mode)) + ': ' + name)
        if len(launched) == 3 and not checks['three_fresh_worker_pids']:
            errors.append('Production repeats lack distinct worker PIDs: ' + str((method, p, N, mode)))
        group_reports.append(dict(method=method, p=p, N=N, timing_mode=mode, records=len(members),
                                  verified=sum(r.get('overall_status') == 'verified_index1' for r in members),
                                  pids=pids, run_ids=[value_of(r, 'run_id') for r in members], **checks))
    launched = [r for r in formal if r.get('monitor', {}).get('worker_pid') is not None]
    hashes = {r.get('code_hash') for r in launched}
    monitor_hashes = {r.get('monitor', {}).get('monitor_code_sha256') for r in launched}
    for r in launched:
        if r.get('overall_status') == 'verified_index1' and r['monitor'].get('worker_exitcode') != 0:
            errors.append('Verified formal worker did not exit zero: '+str(value_of(r,'run_id')))
    expected_hash = prod.get('code_hash')
    current_hash = _current_source_sha256()
    if launched and hashes != {expected_hash}:
        errors.append('Formal numerical version differs from frozen production hash')
    if launched and monitor_hashes != {expected_hash}:
        errors.append('Formal monitor version differs from frozen single-file hash')
    if expected_hash and current_hash != expected_hash:
        errors.append('Current single-file source differs from frozen production hash')
    if prod.get('protocol') is not None and prod['protocol'] != EXPECTED:
        errors.append('Frozen production protocol differs')
    thread_signatures = {json.dumps(r.get('thread_environment'), sort_keys=True) for r in launched}
    if launched and (len(thread_signatures) != 1 or any(not r.get('thread_environment') for r in launched)):
        errors.append('Formal series has missing/mixed thread configuration')
    eig_signatures = {json.dumps(r.get('options', {}).get('eigsh'), sort_keys=True) for r in launched if r.get('options')}
    if len(eig_signatures) > 1:
        errors.append('Formal series mixes eigensolver options')
    if planned:
        modes = ('clean', 'profile') if prod.get('timing_mode') == 'clean_and_profile' else ('profile',)
        expected_groups = {(m, p, N, mode) for mode in modes for m, N in
                           [('pHiSD_H1', n) for n in MANUSCRIPT_GRIDS] + [('standard_HiSD', 128)] for p in (3, 5)}
        planned_groups = {(c.get('method'), c.get('p'), c.get('N'), 'profile' if c.get('profile', True) else 'clean') for c in planned}
        if planned_groups != expected_groups:
            errors.append('Frozen matrix does not contain exact default configurations and measurement modes')
    complete = bool(planned) and not missing and len(formal) == len(planned) and all(g['three_attempt_records'] and g['repetitions_1_2_3'] for g in group_reports)
    fresh = complete and all(g['three_fresh_worker_pids'] and g['all_spawn'] and g['all_workers_exit_zero'] for g in group_reports)
    return dict(integrity_errors=errors, planned_records=len(planned), actual_records=len(formal),
                missing_run_ids=missing, groups=group_reports, formal_matrix_complete=complete,
                all_formal_repeats_fresh=fresh,
                all_formal_records_verified=complete and fresh and all(r.get('overall_status') == 'verified_index1' for r in formal),
                frozen_source_sha256=expected_hash, current_source_sha256=current_hash,
                formal_monitor_code_sha256=sorted(h for h in monitor_hashes if h),
                all_formal_worker_pids_distinct=len(launched) == len({r['monitor']['worker_pid'] for r in launched}))


def independent_endpoint_check(r):
    """Direct five-diagonal operator, independent of solver and Kronecker validator."""
    import numpy as np
    import scipy.sparse as sp
    if not r.get('arrays_path'):
        return dict(executed=False, numerical_outcome=r.get('overall_status'), errors=[])
    path = safe_path(r['arrays_path'])
    before = sha_file(path)
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    result = dict(executed=True, arrays_sha256=before, errors=errors)
    with np.load(path, allow_pickle=False) as z:
        N, p = value_of(r, 'N'), value_of(r, 'p')
        n = N*N
        u = z['u_final']
        check(u.shape == (n,), 'independent endpoint shape mismatch')
        if u.shape != (n,):
            return result
        endpoint_finite = bool(np.isfinite(u).all())
        result['endpoint_finite'] = endpoint_finite
        if not endpoint_finite:
            check(r.get('overall_status') != 'verified_index1', 'verified endpoint is nonfinite')
            return result
        horizontal = -np.ones(n-1, dtype=np.float64)
        horizontal[np.arange(1, n) % N == 0] = 0
        A = sp.diags((-np.ones(n-N), horizontal, np.full(n, 4.), horizontal, -np.ones(n-N)),
                     (-N, -1, 0, 1, N), shape=(n, n), format='csr')
        A.eliminate_zeros()
        h = np.pi/(N+1)
        A = A / h**2
        fresh = float(h*np.linalg.norm(A@u-u**p))
        result['fresh_final_residual'] = fresh
        for key in ('final_residual', 'fresh_final_residual'):
            if finite(r.get(key)):
                check(close(fresh, r[key], 5e-12, 1e-8), 'independent endpoint/' + key + ' mismatch')
        if r.get('solver_status') == 'success':
            hist = z['residual_history']
            check(fresh < 1e-6, 'independent successful residual not below 1e-6')
            check(len(hist) > 0 and bool(np.all(hist[:-1] >= 1e-6)) and hist[-1] < 1e-6,
                  'residual history does not record first solver success')
        if 'eigenvalues' in z and 'eigenvectors' in z:
            lam, Q = z['eigenvalues'], z['eigenvectors']
            result['saved_eigenpair_count'] = len(lam)
            if lam.ndim == 1 and Q.shape == (n, len(lam)) and len(lam):
                H = A-sp.diags(p*u**(p-1), format='csr')
                absolute = np.linalg.norm(H@Q-Q*lam, axis=0)
                relative = absolute/np.maximum(1., np.abs(lam))
                orth = float(np.linalg.norm(Q.T@Q-np.eye(len(lam)), 2))
                complete = bool(r.get('eigensolver_converged') is True and len(lam) == 6)
                accurate = bool(complete and np.isfinite(lam).all() and np.isfinite(absolute).all()
                                and np.all(relative <= 1e-7) and orth <= 1e-8)
                negative_upper = float(lam[0]+10*absolute[0])
                positive_lower = float(lam[1]-10*absolute[1]) if len(lam) > 1 else None
                index_one = bool(accurate and negative_upper < -1e-6 and positive_lower > 1e-6)
                clearly_wrong = bool(accurate and (lam[0]-10*absolute[0] > 1e-6 or lam[1]+10*absolute[1] < -1e-6))
                independent_status = 'verified_index1' if index_one else ('wrong_index' if clearly_wrong else 'verification_unresolved')
                result.update(eigenvalues=lam.tolist(), eigenpair_absolute_residuals=absolute.tolist(),
                              eigenpair_relative_residuals=relative.tolist(), eigenvector_orthogonality_error=orth,
                              negative_upper=negative_upper, positive_lower=positive_lower,
                              independently_classified_cert_status=independent_status)
                check(bool(np.all(np.diff(lam) >= 0)), 'independent eigenvalues not algebraically sorted')
                check(np.array_equal(lam, np.asarray(r.get('eigenvalues', []))), 'independent archive/JSON eigenvalue mismatch')
                if r.get('cert_status') in ('verified_index1', 'wrong_index'):
                    check(r['cert_status'] == independent_status, 'independent certificate classification disagrees')
                if r.get('overall_status') == 'verified_index1':
                    check(index_one and fresh < 1e-6 and r.get('solver_status') == 'success',
                          'overall verification fails independently recomputed acceptance')
                    check(float(np.max(np.abs(np.linalg.norm(Q, axis=0)-1))) <= 1e-8,
                          'independent eigenvectors are not Euclidean unit vectors')
            elif r.get('overall_status') == 'verified_index1':
                errors.append('verified archive lacks six complete eigenpairs')
        elif r.get('overall_status') == 'verified_index1':
            errors.append('verified archive lacks eigenpairs')
    result['raw_file_unchanged'] = sha_file(path) == before
    check(result['raw_file_unchanged'], 'endpoint archive changed during independent validation')
    result['PASS'] = not errors
    return result


def _review_cpu_snapshots(snapshots, pre=False):
    python_sets = [{x['pid'] for x in s.get('busy_processes', [])
                    if 'python' in x.get('comm', '').lower() and x.get('pcpu', 0) > 20} for s in snapshots]
    high = [s.get('system_busy_fraction_estimate', 0) > .8 for s in snapshots]
    persistent = sorted(set.intersection(*python_sets)) if python_sets else []
    pairs = []
    for a, b, x, y in zip(snapshots, snapshots[1:], python_sets, python_sets[1:]):
        if b['unix_time']-a['unix_time'] >= 10 and (x & y or
                (a.get('system_busy_fraction_estimate', 0) > .8 and b.get('system_busy_fraction_estimate', 0) > .8)):
            pairs.append(dict(begin_unix=a['unix_time'], end_unix=b['unix_time'], common_busy_python_pids=sorted(x & y)))
    sustained = bool(len(snapshots) >= 3 and (persistent or all(high))) if pre else bool(pairs)
    return dict(snapshot_count=len(snapshots), persistent_python_pids=persistent,
                sustained_pairs=pairs, sustained_flag=sustained)


def audit_current_trace(r):
    mon = r.get('monitor', {})
    if not mon.get('rss_samples_path'):
        return dict(executed=False, errors=[], warnings=['No launched numerical trace for this outcome.'])
    errors, warnings = [], []
    def check(ok, message):
        if not ok:
            errors.append(message)
    t = r.get('timings', {})
    intervals = r.get('numerical_intervals_ns', {})
    path = safe_path(mon['rss_samples_path'])
    before = sha_file(path)
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    phase_rows = {ph: [] for ph in ('solver', 'cert')}
    for row in rows:
        timestamp = int(row['perf_counter_ns'])
        phase = 'outside_numerical'
        for ph in phase_rows:
            bounds = intervals.get(ph)
            if bounds and bounds[0] <= timestamp <= bounds[1]:
                phase_rows[ph].append(row)
                phase = ph
                break
        if 'phase' in row:
            check(row['phase'] == phase, 'raw RSS phase label disagrees with numerical intervals')
    for ph, key in (('solver', 'T_total'), ('cert', 'T_cert')):
        if ph in intervals:
            a, b = intervals[ph]
            check(a <= b and close((b-a)/1e9, t.get(key), 1e-9, 1e-12), ph+' timer/interval mismatch')
            bounds = mon.get('boundaries_ns', {})
            if bounds:
                check(bounds.get(ph+'_start') == a and bounds.get(ph+'_end') == b, ph+' monitor boundary mismatch')
    if all(ph in intervals for ph in ('solver', 'cert')):
        check(intervals['solver'][1] <= intervals['cert'][0], 'certification overlaps solver total')
    peaks = {ph: max((int(x['rss_bytes']) for x in rs), default=None) for ph, rs in phase_rows.items()}
    full = max((v for v in peaks.values() if v is not None), default=None)
    if intervals:
        check(peaks['solver'] == r.get('solver_peak_rss_bytes'), 'solver sampled peak differs from raw CSV')
        check(full == r.get('full_peak_rss_bytes'), 'full sampled peak differs from raw CSV')
        for key, actual in (('solver_sampled_peak_rss_bytes', peaks['solver']),
                            ('cert_sampled_peak_rss_bytes', peaks['cert']),
                            ('solver_plus_cert_sampled_peak_rss_bytes', full)):
            if key in mon:
                check(mon[key] == actual, key+' disagrees with raw CSV')
    stamps = sorted(int(x['perf_counter_ns']) for rs in phase_rows.values() for x in rs if x.get('source') == 'sample')
    maxgap = max(((b-a)/1e9 for a, b in zip(stamps, stamps[1:])), default=None)
    if 'sample_count' in mon:
        check(mon['sample_count'] == len(rows), 'raw RSS sample count mismatch')
    if intervals and 'numerical_sample_count' in mon:
        check(mon['numerical_sample_count'] == len(stamps), 'raw numerical RSS sample count mismatch')
        check(mon.get('actual_max_numerical_sample_gap_s') == maxgap, 'raw RSS maximum sampling gap mismatch')
    cpu = {}
    for ph, rs in phase_rows.items():
        if ph in intervals and len(rs) < 2:
            warnings.append(ph+' interval has '+str(len(rs))+' RSS samples; transient peaks are unresolved.')
        metric = sorted((x for x in rs if x.get('source') == 'sample' and x.get('cpu_user_ns', '') != ''), key=lambda x: int(x['perf_counter_ns']))
        if len(metric) < 2:
            cpu[ph] = dict(sample_count=len(metric), cpu_seconds_per_wall_second=None)
            check(ph not in mon.get('cpu_phase_samples', {}), ph+' CPU metric reported without enough samples')
            continue
        a, b = metric[0], metric[-1]
        wall = int(b['perf_counter_ns'])-int(a['perf_counter_ns'])
        cpu_ns = sum(int(b[k])-int(a[k]) for k in ('cpu_user_ns', 'cpu_system_ns'))
        runnable_available = all(x.get('runnable_thread_count', '') != '' for x in metric)
        check(runnable_available == mon.get('runnable_thread_count_available', True),
              ph+' runnable-thread availability disagrees with monitor backend')
        values = dict(sample_window_wall_s=wall/1e9, sample_window_cpu_s=cpu_ns/1e9,
                      cpu_seconds_per_wall_second=cpu_ns/wall,
                      max_os_thread_count=max(int(x['thread_count']) for x in metric),
                      max_os_runnable_thread_count=max(int(x['runnable_thread_count']) for x in metric) if runnable_available else None,
                      mean_os_runnable_thread_count=sum(int(x['runnable_thread_count']) for x in metric)/len(metric) if runnable_available else None)
        cpu[ph] = dict(sample_count=len(metric), **values)
        for key, actual in values.items():
            reported = mon.get('cpu_phase_samples', {}).get(ph, {}).get(key)
            check(reported is None if actual is None else close(actual, reported, 5e-12, 0), ph+' raw CPU metric mismatch '+key)
    pre = _review_cpu_snapshots(mon.get('pre_run_contention', {}).get('snapshots', []), True)
    during = _review_cpu_snapshots(mon.get('during_run_cpu_snapshots', []))
    if mon.get('pre_run_contention'):
        check(pre['sustained_flag'] == mon['pre_run_contention'].get('blocked'), 'pre-run contention decision mismatch')
    sustained = pre['sustained_flag'] or during['sustained_flag']
    if value_of(r, 'purpose') == 'production':
        check(not sustained, 'formal timing has sustained contention requiring review')
    if r.get('overall_status') == 'verified_index1':
        post = r.get('postprocessing_intervals_ns', {})
        for label in ('structure_export', 'array_save'):
            if label in post:
                a, b = post[label]
                check(a >= intervals.get('cert', [0, 0])[1] and b >= a, label+' overlaps numerical intervals')
        if all(label in post for label in ('structure_export', 'array_save')):
            check(post['structure_export'][1] <= post['array_save'][0], 'array save precedes structural export completion')
        events = mon.get('events', [])
        names = [e.get('event') for e in events]
        expected = ['baseline', 'solver_end', 'cert_start', 'cert_end', 'done']
        check([x for x in names if x in expected] == expected, 'worker numerical/export event order mismatch')
        ed = {e.get('event'): e for e in events}
        if all(x in ed for x in expected):
            check(all(k not in ed['cert_end'].get('record', {}) for k in ('structure', 'arrays_path', 'arrays_sha256')),
                  'postprocessing evidence exists before numerical certification ends')
            check(all(k in ed['done'].get('record', {}) for k in ('structure', 'arrays_path', 'arrays_sha256')),
                  'worker completion lacks postprocessing evidence')
    check(sha_file(path) == before, 'raw RSS/CPU trace changed during validation')
    return dict(executed=True, errors=errors, warnings=warnings, PASS=not errors,
                rss_samples_sha256=before, numerical_intervals_ns=intervals,
                phase_sample_counts={k: len(v) for k, v in phase_rows.items()}, sampled_peaks_bytes=peaks,
                actual_max_numerical_sample_gap_s=maxgap, cpu_phase_recomputed=cpu,
                pre_run_contention=pre, during_run_contention=during, sustained_contention_detected=sustained,
                cumulative_highwater_bytes={k: r.get(k) for k in ('solver_start_highwater_bytes', 'solver_end_highwater_bytes', 'cert_end_highwater_bytes')})


def overhead_validation(records, first, recheck, decision, prod):
    errors = []
    expected = {('pHiSD_H1', N, p) for N in (64, 256) for p in (3, 5)} | {('standard_HiSD', 128, p) for p in (3, 5)}
    key = lambda g: (g.get('method'), g.get('N'), g.get('p'))
    by_id = {value_of(r, 'run_id'): r for r in records}
    if not first or not decision:
        return dict(PASS=False, errors=['Missing current overhead audit/decision'], effective_pairs=0)
    first_groups = first.get('groups', [])
    if first.get('status') != 'PASS' or len(first_groups) != 6 or {key(g) for g in first_groups} != expected:
        errors.append('First overhead audit lacks six passing configuration groups')
    candidate_groups = {key(g): g for g in first_groups}
    if recheck:
        if recheck.get('status') != 'PASS':
            errors.append('Finite overhead recheck failed')
        candidate_groups.update({key(g): g for g in recheck.get('groups', [])})
    groups = decision.get('groups', [])
    if len(groups) != 6 or {key(g) for g in groups} != expected:
        errors.append('Final overhead decision configuration coverage mismatch')
    sustained_flags, pair_count = [], 0
    for group in groups:
        if group != candidate_groups.get(key(group)):
            errors.append('Final overhead group differs from retained source audit')
        ratios = []
        pairs = group.get('pairs', [])
        if len(pairs) != 3 or {x.get('pair') for x in pairs} != {1, 2, 3}:
            errors.append('Overhead group lacks three unique pairs')
        for pair in pairs:
            raw = {label: by_id.get(pair.get(label+'_run')) for label in ('profile', 'clean')}
            if any(r is None for r in raw.values()):
                errors.append('Overhead decision references missing current raw record')
                continue
            for label, r in raw.items():
                if (value_of(r, 'method'), value_of(r, 'N'), value_of(r, 'p')) != key(group) or mode_of(r) != label:
                    errors.append('Overhead raw pair configuration/mode mismatch')
                if r.get('overall_status') != 'verified_index1':
                    errors.append('Overhead pair includes nonverified outcome')
                if not close(r.get('timings', {}).get('T_total'), pair.get(label+'_total'), 1e-12, 1e-12):
                    errors.append('Overhead pair timing differs from raw record')
            if any(config_of(raw['profile']).get(k) != config_of(raw['clean']).get(k) for k in ('seed_init', 'seed_cert', 'protocol')):
                errors.append('Overhead pair inputs differ')
            tp = raw['profile'].get('timings', {}).get('T_total')
            tc = raw['clean'].get('timings', {}).get('T_total')
            if finite(tp) and finite(tc) and tc > 0:
                ratio = tp/tc-1
                ratios.append(ratio)
                if not close(ratio, pair.get('overhead_fraction'), 1e-12, 1e-12):
                    errors.append('Overhead paired ratio differs from raw record')
            if pair.get('order') != (['profile', 'clean'] if pair.get('pair', 0) % 2 else ['clean', 'profile']):
                errors.append('Overhead paired order was not interleaved')
            pair_count += 1
        if len(ratios) == 3:
            med = statistics.median(ratios)
            sustained = med > .05 and sum(v > .05 for v in ratios) >= 2
            sustained_flags.append(sustained)
            if not close(med, group.get('median_overhead_fraction'), 1e-12, 1e-12) or group.get('sustained_over_5pct') is not sustained:
                errors.append('Overhead sustained >5% decision differs from raw pairs')
    for report in (first, recheck):
        for group in (report or {}).get('groups', []):
            for pair in group.get('pairs', []):
                if any(pair.get(label+'_run') not in by_id for label in ('profile', 'clean')):
                    errors.append('An original overhead raw record was discarded')
    if len(sustained_flags) == 6 and decision.get('use_clean_production') is not any(sustained_flags):
        errors.append('Production timer decision differs from sustained overhead rule')
    if prod and prod.get('timing_mode') != ('clean_and_profile' if decision.get('use_clean_production') else 'profile'):
        errors.append('Frozen production modes differ from overhead decision')
    return dict(PASS=not errors and pair_count == 18, errors=errors, effective_pairs=pair_count,
                finite_recheck_performed=bool(recheck), use_clean_production=decision.get('use_clean_production'))


def _audit_summary_file(name, records, select):
    formal = [r for r in records if value_of(r, 'purpose') == 'production']
    by_id = {value_of(r, 'run_id'): r for r in formal}
    expected_groups = {(value_of(r, 'method'), value_of(r, 'p'), value_of(r, 'N'), mode_of(r)) for r in formal if select(r)}
    with (BASE/name).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    observed, errors, fields_checked = [], [], 0
    for row in rows:
        group = (row['method'], int(row['p']), int(row['N']), row['timing_mode'])
        observed.append(group)
        ids = json.loads(row['run_ids'])
        expected_ids = {value_of(r, 'run_id') for r in formal if (value_of(r, 'method'), value_of(r, 'p'), value_of(r, 'N'), mode_of(r)) == group}
        if set(ids) != expected_ids or len(ids) != len(expected_ids):
            errors.append('Summary membership mismatch: '+str(group))
            continue
        for col, value in row.items():
            if col.startswith('observed_max_all_outcomes_'):
                field = col.removeprefix('observed_max_all_outcomes_')
                numbers = [by_id[rid].get(field) for rid in ids if finite(by_id[rid].get(field))]
                expected = max(numbers) if numbers else None
                fields_checked += 1
                if not (value == '' if expected is None else value != '' and float(value) == expected):
                    errors.append('Summary all-outcome maximum mismatch: '+str(group)+' '+col)
                continue
            textcol = col.removeprefix('solver_success_')
            prefix, _, field = textcol.partition('_')
            if prefix not in ('median', 'min', 'max', 'count') or not field:
                continue
            subset = [by_id[rid] for rid in ids if (by_id[rid].get('solver_status') == 'success' if col.startswith('solver_success_') else by_id[rid].get('overall_status') == 'verified_index1')]
            if row.get('mixed_source_statistics_withheld') == 'True':
                subset = []
            numbers = []
            for r in subset:
                merged = dict(r)
                merged.update(r.get('timings', {})); merged.update(r.get('counters', {})); merged.update(r.get('structure', {}))
                if finite(merged.get(field)):
                    numbers.append(merged[field])
            expected = len(numbers) if prefix == 'count' else ({'median': statistics.median, 'min': min, 'max': max}[prefix](numbers) if numbers else None)
            fields_checked += 1
            if not (value == '' if expected is None else value != '' and float(value) == expected):
                errors.append('Summary statistic mismatch: '+str(group)+' '+col)
        if int(row['record_count']) != len(ids) or int(row['verified_repeat_count']) != sum(by_id[rid].get('overall_status') == 'verified_index1' for rid in ids):
            errors.append('Summary outcome denominator mismatch: '+str(group))
    if set(observed) != expected_groups or len(observed) != len(set(observed)):
        errors.append('Summary group coverage/duplicate mismatch: '+name)
    return dict(file=name, PASS=not errors, errors=errors, numeric_fields_checked=fields_checked)


def run_final_validation():
    start = time.perf_counter()
    records, rows, per_run, errors = [], [], [], []
    for path in sorted((BASE/'runs').glob('*.monitor.json')):
        try:
            r = normalize_record(read_json(path))
            records.append(r); rows.append(flatten(r, str(path)))
            err = validate_record(r)
            detail = dict(run_id=value_of(r, 'run_id'), purpose=value_of(r, 'purpose'), overall_status=r.get('overall_status'), errors=err,
                          warnings=record_warnings(r), raw_record_sha256=sha_file(path))
            if value_of(r, 'purpose') == 'production':
                detail['independent_endpoint'] = independent_endpoint_check(r)
                detail['trace_audit'] = audit_current_trace(r)
                err.extend(detail['independent_endpoint']['errors']); err.extend(detail['trace_audit']['errors'])
                detail['warnings'].extend(detail['trace_audit']['warnings'])
            detail['integrity_pass'] = not err
            per_run.append(detail)
            errors.extend(path.name+': '+e for e in err)
        except Exception as ex:
            errors.append(path.name+': '+type(ex).__name__+': '+str(ex))
    optional = lambda name: read_json(BASE/name) if (BASE/name).is_file() else None
    prod = optional('production_config.json') or {}
    production = production_check(records, prod)
    errors.extend(production['integrity_errors'])
    overhead = overhead_validation(records, optional('overhead_report.json'), optional('overhead_check_report.json'), optional('overhead_decision.json'), prod)
    errors.extend(overhead['errors'])
    summaries = summarize(rows)
    write_csv('raw_runs.csv', rows)
    write_csv('scalability_summary.csv', [s for s in summaries if s['method'] == 'pHiSD_H1'])
    write_csv('fixed_grid_cost_summary.csv', [s for s in summaries if s['N'] == 128])
    write_csv('cost_breakdown.csv', summaries)
    memcols = ['method', 'p', 'N', 'n', 'timing_mode', 'record_count', 'verified_repeat_count', 'solver_success_count',
               'failed_or_censored_count', 'status_counts', 'run_ids', 'code_hashes']
    memcols += list(dict.fromkeys(k for s in summaries for k in s if any(k.endswith(m) for m in MEMORY+STRUCTURE)))
    write_csv('memory_structure.csv', [{k: s.get(k) for k in memcols} for s in summaries], memcols)
    summary_audits = [_audit_summary_file(name, records, select) for name, select in
                      [('scalability_summary.csv', lambda r: value_of(r, 'method') == 'pHiSD_H1'),
                       ('fixed_grid_cost_summary.csv', lambda r: value_of(r, 'N') == 128),
                       ('cost_breakdown.csv', lambda r: True), ('memory_structure.csv', lambda r: True)]]
    for audit in summary_audits:
        errors.extend(audit['errors'])
    regression = optional('REGRESSION_REPORT.json') or {}
    validator_tests = optional('VALIDATOR_TEST_REPORT.json') or {}
    gate2 = optional('gate2_dry_validation.json') or {}
    regression_pass = regression.get('PASS') is True
    validator_tests_pass = validator_tests.get('PASS') is True
    archive_tests = optional('ARCHIVE_SAFETY_TEST_REPORT.json') or {}
    archive_tests_pass = archive_tests.get('PASS') is True
    gate2_pass = gate2.get('status') == 'PASS'
    for label, report in [('regression', regression), ('validator_tests', validator_tests), ('gate2', gate2),
                          ('overhead', optional('overhead_report.json') or {}), ('overhead_decision', optional('overhead_decision.json') or {})]:
        if report.get('code_hash') and prod.get('code_hash') and report['code_hash'] != prod['code_hash']:
            errors.append(label+' report source hash differs from formal source')
    complete = production['formal_matrix_complete'] and production['all_formal_repeats_fresh'] and production['all_formal_records_verified']
    ready = not errors and complete and regression_pass and validator_tests_pass and gate2_pass and overhead['PASS'] and archive_tests_pass
    warnings = [dict(run_id=d['run_id'], warning=w) for d in per_run for w in d['warnings']]
    report = dict(schema='single_file_current_run_validation_v1', generated_utc=datetime.now(timezone.utc).isoformat(),
                  data_integrity_pass=not errors, all_automatic_validation_pass=ready,
                  execution_status=('COMPLETE_WITH_LIMITATIONS' if warnings else 'COMPLETE') if ready else 'BLOCKED',
                  evidence_status='READY' if ready else 'INSUFFICIENT',
                  production=production, per_run=per_run, summary_audits=summary_audits,
                  overhead_validation=overhead, regression_pass=regression_pass, validator_tests_pass=validator_tests_pass,
                  archive_tests_pass=archive_tests_pass, gate2_dry_pass=gate2_pass, overhead_pass=overhead['PASS'], errors=errors, warnings=warnings,
                  independent_validation=dict(saved_current_endpoints=True, direct_five_diagonal_operator=True, eigensolves=0,
                                              historical_results_read=False, acceptance_not_relaxed=True),
                  elapsed_wall_s=time.perf_counter()-start)
    (BASE/'validation_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def generate_outputs(validation, run_metadata=None):
    metadata = _publication_data(run_metadata or {})
    metadata['result_directory'] = 'outputs/7.5/7.5.2'
    records = [normalize_record(read_json(p)) for p in sorted((BASE/'runs').glob('*.monitor.json'))]
    summaries = summarize([flatten(r) for r in records])
    formal = [r for r in records if value_of(r, 'purpose') == 'production']
    def formatted(s, key, divisor=1):
        vals = [s.get(stat+'_'+key) for stat in ('median', 'min', 'max')]
        return '—' if any(v is None for v in vals) else f'{vals[0]/divisor:.5g} [{vals[1]/divisor:.5g}, {vals[2]/divisor:.5g}]'
    lines = ['# Lane–Emden scalability results', '',
             f"Execution: **{validation.get('execution_status', 'BLOCKED')}**.", '',
             f"Default formal matrix: p-HiSD N=64,128,192,256 for p=3,5; standard HiSD N=128 for p=3,5. Three fresh workers per configuration and per selected timing mode.",
             f"Recorded formal attempts: {len(formal)}; verified outcomes: {sum(r.get('overall_status') == 'verified_index1' for r in formal)}.",
             f"Current-run metadata (elapsed values here are measured before final cleanup, not the final whole-workflow wall time): `{json.dumps(metadata, ensure_ascii=False, sort_keys=True)}`", '',
             '| Method | p | N | Mode | Verified/attempts | Solver s | Certification s | Solver+cert s | Sampled full RSS MiB |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append('| '+' | '.join([s['method'], str(s['p']), str(s['N']), s['timing_mode'],
                    f"{s['verified_repeat_count']}/{s['record_count']}", formatted(s, 'T_total'), formatted(s, 'T_cert'),
                    formatted(s, 'T_total_with_cert'), formatted(s, 'full_peak_rss_bytes', 2**20)])+' |')
    lines += ['', 'Entries show median [min, max] among verified runs; failed/censored attempts remain in outcome denominators; temporary raw evidence is checked before cleanup. Different timing modes are separate series.', '',
              '## Measurement interpretation', '',
              '- T_total is the solver wall interval, including operator assembly, setup, initialization and updates. T_cert lies outside T_total. Numerical T_total_with_cert is their sum; whole workflow wall time also includes imports, gates, monitoring, saves, verification and final cleanup.',
              '- T_eig_inclusive overlaps M apply/solve. The additive per-run view is setup/update + apply/solve + eig exclusive + other. Independent column medians need not sum to median total; these marginal medians are never plotted as an additive stack.',
              '- Sampled RSS uses only exact numerical intervals. Missing short-interval samples remain null. Cumulative OS high-water is separately retained and is never subtracted to infer stage memory. Exported sparse-array bytes exclude unexported native-library workspaces.',
              '- LU export, NPZ compression, file output and plotting occur after numerical measurement. Requested RSS sampling is approximately 20 ms; raw gaps and sample counts are checked before temporary traces are removed; sampling limitations are listed below.', '',
              '## Scope and limitations', '',
              'The measured implementation fixes k=1, J=5 and sparse LU. With n=N² and F_n=nnz(L)+nnz(U), per-update work includes O(Jn+(J+1)F_n); setup, factorization, initial eigensolve and certification add separate cost. Main persistent storage includes O(n+F_n+n*ncv), while sampled RSS covers observed process/native memory. These observations establish neither linear overall complexity nor mesh-independent convergence, high-index or parallel scaling, performance at unrun N=768/1024, or independent optimal tuning of the comparison method.', '',
              'Certification independently checks saved endpoints and six ordinary full-space eigenpairs, residuals, orthogonality and resolved index-one sign margins. It is a numerical acceptance protocol, not an interval-arithmetic proof.', '',
              '## Automatic acceptance', '',
              f"regression_pass={validation.get('regression_pass')}; validator_tests_pass={validation.get('validator_tests_pass')}; gate2_dry_pass={validation.get('gate2_dry_pass')}; overhead_pass={validation.get('overhead_pass')}; data_integrity_pass={validation.get('data_integrity_pass')}; all_automatic_validation_pass={validation.get('all_automatic_validation_pass')}.", '']
    for error in validation.get('errors', []):
        lines.append('- Validation error: '+error)
    failures = [r for r in formal if r.get('overall_status') != 'verified_index1']
    for r in failures:
        lines.append('- '+str(value_of(r, 'run_id'))+': '+str(r.get('overall_status'))+'; '+str(r.get('reason') or (r.get('resource_decision') or {}).get('reason') or r.get('monitor', {}).get('stop_reason') or r.get('solver_exception') or r.get('cert_exception')))
    for warning in validation.get('warnings', []):
        if isinstance(warning, dict):
            lines.append('- Sampling qualification ' + str(warning.get('run_id', ''))
                         + ': ' + str(warning.get('warning', warning)))
        else:
            lines.append('- Sampling qualification: ' + str(warning))
    production = validation.get('production') or {}
    lines.append('Requested formal matrix complete: ' + str(production.get('formal_matrix_complete', False)) + '.')
    for run_id in production.get('missing_run_ids', []):
        lines.append('- Missing requested run: ' + str(run_id))
    lines += ['', '## Retained output files', '',
              'Only scalability_summary.csv, fixed_grid_cost_summary.csv, cost_breakdown.csv, memory_structure.csv and this overview are retained after successful validation. Temporary raw records, monitoring traces and internal reports are used for validation and statistics, then removed. No evidence ZIP, plots or separate logs are kept.',
              'If the run fails, available tables are explicitly incomplete and RUN_FAILURE.txt records the stage, configuration and reason.', '']
    (BASE/'RESULTS_OVERVIEW.md').write_text(_publication_data('\n'.join(lines)+'\n'), encoding='utf-8')
    return dict(summary_groups=len(summaries), formal_records=len(formal), plots=[])


def run_archive_selftests():
    return run_archive_safety_tests()


def preflight_current_run():
    if code_hash() != IMPORTED_SOURCE_SHA256:
        raise RuntimeError('Loaded source changed before environment preflight')
    source_dir=BASE/'source';source_dir.mkdir()
    with (source_dir/'run.py').open('xb') as handle:
        handle.write(SOURCE_FILE.read_bytes())
    info=environment_info()
    config_buffer=io.StringIO()
    with contextlib.redirect_stdout(config_buffer):
        np.show_config()
    info.update(numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
                numpy_build_config=config_buffer.getvalue(),code_hash=code_hash(),
                required_dependencies=REQUIRED_DEPENDENCIES,
                installed_packages_automatically=False,
                source_imported_sha256=IMPORTED_SOURCE_SHA256,
                single_file_entry=str(SOURCE_FILE),output_root=str(OUTPUT_ROOT),result_directory=str(BASE))
    write_json(BASE/'environment.json',info)
    write_json(BASE/'resource_budget.json',dict(start_unix=BOOT_START_UNIX,active_wall_budget_s=10800,
        code_hash=code_hash(),semantics='Elapsed wall from this invocation before dependency imports; includes all gates, imports, monitoring and postprocessing. Never extended automatically.'))
    write_json(BASE/'startup_configuration.json',dict(code_hash=code_hash(),protocol=PROTOCOL,
        requested_phisd_grids=list(MANUSCRIPT_GRIDS),requested_standard_grids=[128],p_values=[3,5],repetitions=3,
        no_arguments_required=True,max_N=max(MANUSCRIPT_GRIDS),solver_timeout_s=900,cert_timeout_s=900,
        overhead_rule='Three interleaved profile/clean pairs at all six audited configurations. Sustained >5% uses separate clean and profile formal series. One finite group recheck is predeclared.',
        source_frozen_before_tests=True,all_workers_spawn=True))
    if info['disk_free_bytes'] < 512*2**20:
        raise RuntimeError('Output disk has less than 512 MiB free; existing results are preserved')
    if info['rss_limit_bytes'] < 64*2**20:
        raise RuntimeError('Current memory budget cannot safely start numerical checks')
    if rss_bytes() is None:
        raise RuntimeError('Native RSS monitoring is unavailable')
    gate_log('Environment/source/resource preflight PASS')
    return info


def main():
    mp.freeze_support()
    if BASE is None:
        raise RuntimeError('Run this file directly to allocate a new result directory')
    initialize_archive_ownership(BASE, created_current_run=True)
    failure = None
    validation = None
    stage = 'preflight'
    metadata = dict(source=str(SOURCE_FILE), source_sha256=IMPORTED_SOURCE_SHA256,
        result_directory=str(BASE), started_unix=BOOT_START_UNIX,
        historical_results_required=False)
    try:
        preflight_current_run()
        stage = 'output_cleanup_selftests'
        archive_test_report = run_archive_selftests()
        if not archive_test_report.get('PASS'):
            raise RuntimeError('Output retention/cleanup safety selftests failed')
        stage = 'regression'
        regression = run_regression()
        if not regression['PASS']:
            raise RuntimeError('Fresh regression failed; dependent experiments were not started')
        if code_hash() != IMPORTED_SOURCE_SHA256:
            raise RuntimeError('Source changed during regression')
        stage = 'smoke N=16 p=3'
        smoke = gate_case(16, 3, 'smoke')
        gate_validate(smoke)
        stage = 'validator_selftests'
        validator = run_validator_tests(smoke)
        if not validator['PASS']:
            raise RuntimeError('Fresh validator tests failed; dependent experiments were not started')
        stage = 'dry_run'
        gate_dry()
        stage = 'overhead'
        gate_overhead()
        stage = 'configuration_freeze'
        gate_freeze()
        stage = 'production'
        gate_production()
    except BaseException as exc:
        failure = _output_failure_details(exc, stage)
        traceback.print_exc()
    try:
        stage = 'final_validation'
        gate_log('Independent endpoint, timing, RSS, counters and source validation started')
        validation = run_final_validation()
        if failure:
            validation['all_automatic_validation_pass'] = False
            validation['execution_status'] = 'BLOCKED'
            validation['evidence_status'] = 'INSUFFICIENT'
            validation['workflow_failure'] = failure
        if code_hash() != IMPORTED_SOURCE_SHA256:
            validation['all_automatic_validation_pass'] = False
            validation['execution_status'] = 'BLOCKED'
            validation['errors'].append('Source changed after import')
        metadata['whole_run_pre_cleanup_elapsed_s'] = (CLOCK()-BOOT_START_NS)/1e9
        metadata['elapsed_measurement_scope'] = 'Pre-cleanup elapsed includes imports/gates/production/validation. Whole-run elapsed is printed after final output cleanup.'
        stage = 'result_summary'
        generate_outputs(validation, metadata)
        gate_log('Automatic validation=' + str(validation['all_automatic_validation_pass'])
                 + '; retaining only five key output files')
    except BaseException as exc:
        finalization_failure = _output_failure_details(exc, stage)
        traceback.print_exc()
        if failure is None:
            failure = finalization_failure
        if validation is None:
            validation = dict(all_automatic_validation_pass=False, errors=[str(exc)])
        validation['all_automatic_validation_pass'] = False
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    success = bool(not failure and validation and validation.get('all_automatic_validation_pass'))
    if not success and failure is None:
        production = (validation or {}).get('production') or {}
        failure = dict(stage='final_validation',
            configuration={'missing_run_ids': production.get('missing_run_ids', [])},
            message='; '.join(str(e) for e in ((validation or {}).get('errors') or []))
                    or 'Required validation or formal-matrix completion checks did not pass.')
    transfer = None
    result_directory = BASE
    try:
        transfer_owner = dict(_ARCHIVE_RUN_OWNERS[BASE.resolve()]) if success else None
        cleanup = archive_evidence(success, failure=failure)
    except BaseException as exc:
        success = False
        cleanup = None
        try:
            directory = _owned_output_directory()
            _write_failure_outputs(directory, _output_failure_details(exc, 'output_cleanup'))
        except BaseException:
            pass
        print('Output cleanup could not finish safely: ' + str(exc), file=sys.stderr, flush=True)
    if success:
        transfer = transfer_final_outputs(transfer_owner, cleanup)
        success = transfer['success']
        result_directory = transfer['result_directory']
    elapsed = (CLOCK()-BOOT_START_NS)/1e9
    final = dict(status='PASS' if success else 'BLOCKED', result_directory=str(result_directory),
        whole_run_wall_seconds=elapsed, source_sha256=IMPORTED_SOURCE_SHA256,
        regression_pass=bool(validation and validation.get('regression_pass')),
        validation_pass=bool(validation and validation.get('all_automatic_validation_pass')),
        output_cleanup=cleanup, output_transfer=transfer)
    print('RUN_COMPLETED ' + json.dumps(_publication_data(final), ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
