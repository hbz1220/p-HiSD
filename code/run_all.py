#!/usr/bin/env python3
# Run the ten Section 7 experiments in paper order with their default parameters.
# Each experiment writes to its corresponding outputs/ directory.

from __future__ import annotations

import argparse
from datetime import datetime
import errno
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time


CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
IS_WINDOWS = sys.platform == "win32"
# MATLAB generates the data for 7.4 and 7.5.1 before Python draws their figures.
SECTIONS = {
    "7.1": ("7.1/run.py",),
    "7.2.1": ("7.2/7.2.1/run.py",),
    "7.2.2": ("7.2/7.2.2/run.py",),
    "7.3": ("7.3/run.py",),
    "7.4": ("7.4/run.m", "7.4/fig.py"),
    "7.5.1": ("7.5/7.5.1/run.m", "7.5/7.5.1/fig.py"),
    "7.5.2": ("7.5/7.5.2/run.py",),
    "7.5.3": ("7.5/7.5.3/run.py",),
    "7.5.4": ("7.5/7.5.4/run.py",),
    "7.6": ("7.6/run.py",),
}


def find_python(requested, sections):
    packages = ["numpy", "scipy", "matplotlib"]
    if "7.2.2" in sections:
        packages.append("sympy")
    if any(section in sections for section in ("7.3", "7.5.4")):
        packages.append("threadpoolctl")
    relative_python = "Scripts/python.exe" if IS_WINDOWS else "bin/python"
    candidates = ([requested] if requested else
                  [ROOT / ".shared_venv" / relative_python,
                   ROOT / ".venv" / relative_python, sys.executable])
    failures = []
    for candidate in candidates:
        executable = shutil.which(str(candidate))
        if executable is None:
            failures.append(f"{candidate}: not found")
            continue
        probe = "import sys; assert sys.version_info >= (3, 10); import " + ", ".join(packages)
        if "7.5.2" in sections:
            probe += ("\nif sys.platform != 'darwin':\n"
                      " import psutil\n"
                      " assert tuple(map(int, psutil.__version__.split('.')[:2])) >= (5, 9), '7.5.2 requires psutil >=5.9'\n")
        try:
            result = subprocess.run([executable, "-c", probe], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    env=python_environment(), timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{candidate}: {exc}")
            continue
        if result.returncode == 0:
            return executable
        failures.append(f"{candidate}: {result.stderr.strip()}")
    raise RuntimeError("No Python environment with required dependencies. Use --python PATH.\n"
                       + "\n".join(failures))


def find_matlab(requested):
    if IS_WINDOWS:
        locations = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "MATLAB"]
        pattern = "R*/bin/matlab.exe"
    elif sys.platform == "darwin":
        locations, pattern = [Path("/Applications")], "MATLAB*.app/bin/matlab"
    else:
        locations, pattern = [Path("/usr/local/MATLAB"), Path("/opt/MATLAB")], "R*/bin/matlab"
    candidates = ([requested] if requested else
                  [os.environ.get("MATLAB_EXECUTABLE"), shutil.which("matlab"),
                   *sorted((p for root in locations for p in root.glob(pattern)), reverse=True)])
    for candidate in candidates:
        if candidate:
            executable = shutil.which(str(candidate))
            if executable:
                return executable
    raise RuntimeError("MATLAB is required for 7.4 and 7.5.1. Use --matlab /path/to/bin/matlab.")


def make_plan(sections, python, matlab):
    plan = []
    for section in sections:
        for relative in SECTIONS[section]:
            script = CODE / relative
            if not script.is_file():
                raise RuntimeError(f"Missing script: {script}")
            if script.suffix == ".m":
                expression = "run('" + script.as_posix().replace("'", "''") + "')"
                command = [matlab, "-batch", expression]
            else:
                command = [python, "-u", str(script)]
            plan.append(dict(section=section, script=relative, command=command, status="PENDING"))
    if "7.5.4" in sections and not (CODE / "7.5/7.5.4/LDG_inputs.npz").is_file():
        raise RuntimeError("Missing 7.5.4 input: LDG_inputs.npz")
    return plan


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


def save_status(directory, state):
    temporary = directory / "status.json.tmp"
    temporary.write_text(json.dumps(_publication_data(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "status.json")


def python_environment():
    return dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg")


def format_command(command):
    return subprocess.list2cmdline(command) if IS_WINDOWS else shlex.join(command)


def lock_launcher(lock):
    if IS_WINDOWS:
        import msvcrt
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise BlockingIOError("Launcher lock is already held") from exc
            raise
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


class WindowsJob:

    def __init__(self):
        import ctypes
        from ctypes import wintypes as w

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", w.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_processes", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class IOCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IOCounters),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            ("SetInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            ("OpenProcess", [w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            ("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL),
            ("CloseHandle", [w.HANDLE], w.BOOL),
        ):
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, pid):
        import ctypes
        handle = self.api.OpenProcess(0x0100 | 0x0001, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.api.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.api.CloseHandle(handle)

    def close(self):
        if self.handle:
            import ctypes
            if not self.api.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None


def start_windows_step(command, options):
    wrapper = ("import subprocess, sys; "
               "ready = sys.stdin.buffer.read(1); "
               "sys.exit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL) if ready == b'1' else 1)")
    job = WindowsJob()
    process = None
    try:
        process = subprocess.Popen([sys.executable, "-u", "-c", wrapper, *command],
                                   stdin=subprocess.PIPE,
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, **options)
        job.attach(process.pid)
        process.stdin.write("1")
        process.stdin.flush()
        process.stdin.close()
        return process, job
    except BaseException:
        try:
            job.close()
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
        raise


def stop_group(process, job=None):
    if IS_WINDOWS:
        try:
            if process.poll() is None:
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        finally:
            job.close()
            process.wait(timeout=5)
        return
    for sig, wait in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGINT:
            continue
        break
    process.wait()


def run_step(command, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        options = dict(cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace", bufsize=1,
                       env=python_environment())
        job = None
        if IS_WINDOWS:
            process, job = start_windows_step(command, options)
        else:
            process = subprocess.Popen(command, start_new_session=True, **options)
        try:
            for line in process.stdout:
                line = _publication_data(line)
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                stop_group(process, job)
            finally:
                signal.signal(signal.SIGINT, previous)
            raise
        finally:
            process.stdout.close()
            if job is not None:
                job.close()


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sections", nargs="+", choices=list(SECTIONS))
    selection.add_argument("--from-section", choices=list(SECTIONS))
    parser.add_argument("--dry-run", action="store_true", help="check dependencies and list commands; do not run experiments")
    parser.add_argument("--python-only", action="store_true", help="skip MATLAB sections 7.4 and 7.5.1, including their dependent plots")
    parser.add_argument("--python", help="Python interpreter path")
    parser.add_argument("--matlab", help="MATLAB executable path")
    args = parser.parse_args()
    sections = list(SECTIONS)
    if args.sections:
        sections = [s for s in sections if s in args.sections]
    elif args.from_section:
        sections = sections[sections.index(args.from_section):]
    if args.python_only:
        skipped = [s for s in sections if any(Path(p).suffix == ".m" for p in SECTIONS[s])]
        sections = [s for s in sections if s not in skipped]
        if not sections:
            parser.error("No Python-only sections remain in this selection")
        if skipped:
            print("Skipping MATLAB-dependent sections: " + ", ".join(skipped))
    try:
        python = find_python(args.python, sections)
        matlab = find_matlab(args.matlab) if any(s in sections for s in ("7.4", "7.5.1")) else None
        plan = make_plan(sections, python, matlab)
    except RuntimeError as exc:
        parser.error(str(exc))
    print(f"Python: {python}")
    if matlab:
        print(f"MATLAB: {matlab}")
    for index, step in enumerate(plan, 1):
        print(f"{index:02d}. [{step['section']}] {format_command(step['command'])}")
    if args.dry_run:
        print("Preflight passed. No experiments run." +
              (" MATLAB license availability is checked when MATLAB starts." if matlab else ""))
        return 0
    log_root = ROOT / "outputs/run_all"
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / ".launcher.lock").open("a+b") as lock:
        try:
            lock_launcher(lock)
        except BlockingIOError:
            parser.error("Another run_all.py is already running in this repository.")
        except OSError as exc:
            parser.error(f"Cannot lock the launcher: {exc}")
        directory = log_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory.mkdir()
        state = dict(status="RUNNING", sections=sections, steps=plan,
                     started=datetime.now().isoformat(),
                     meaning="Process exit status only; experiment records determine scientific outcomes.")
        save_status(directory, state)
        print(f"Sequential run; logs: {directory}", flush=True)
        for index, step in enumerate(plan, 1):
            step.update(status="RUNNING", log=f"{index:02d}_{step['section']}_{Path(step['script']).stem}.log")
            save_status(directory, state)
            started = time.monotonic()
            print(f"\n[{index}/{len(plan)}] {step['script']}", flush=True)
            try:
                code = run_step(step["command"], directory / step["log"])
                step.update(exit_code=code, status="COMPLETED" if code == 0 else "FAILED")
            except KeyboardInterrupt:
                step.update(status="INTERRUPTED", exit_code=130)
            except Exception as exc:
                step.update(status="FAILED", exit_code=1, error=str(exc))
            step["elapsed_seconds"] = time.monotonic() - started
            # Stop after a failed process; scientific outcomes remain in each experiment's records.
            if step["status"] != "COMPLETED":
                state.update(status=step["status"], finished=datetime.now().isoformat())
                save_status(directory, state)
                print(f"Stopped: {step['script']}. See {directory / step['log']}")
                print(f"After resolving the issue: python code/run_all.py --from-section {step['section']}")
                return step["exit_code"] if step["exit_code"] > 0 else 1
            save_status(directory, state)
        state.update(status="COMPLETED", finished=datetime.now().isoformat())
        save_status(directory, state)
        print(f"All {len(plan)} commands completed. Status: {directory / 'status.json'}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
