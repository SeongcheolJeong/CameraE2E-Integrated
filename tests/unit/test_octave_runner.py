from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pyisetcam.octave_runner import (
    find_octave_binary,
    latest_octave_crash_log,
    octave_startup_env,
    run_octave,
)


def _make_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


def test_find_octave_binary_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "octave-cli-10.3.0"
    _make_executable(explicit)

    resolved = find_octave_binary(preferred=explicit)

    assert resolved == explicit


def test_find_octave_binary_prefers_versioned_cli_over_wrapper(tmp_path: Path) -> None:
    wrapper = tmp_path / "octave"
    versioned = tmp_path / "octave-cli-10.3.0"
    _make_executable(wrapper)
    _make_executable(versioned)

    resolved = find_octave_binary(search_paths=[tmp_path])

    assert resolved == versioned


def test_latest_octave_crash_log_returns_newest_match(tmp_path: Path) -> None:
    older = tmp_path / "octave-cli-10.3.0-older.ips"
    newer = tmp_path / "octave-cli-10.3.0-newer.ips"
    older.write_text("older")
    newer.write_text("newer")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    resolved = latest_octave_crash_log(diagnostics_dir=tmp_path, prefix="octave-cli")

    assert resolved == newer


def test_octave_startup_env_derives_conda_runtime_paths(tmp_path: Path) -> None:
    runtime_root = tmp_path / "env"
    binary = runtime_root / "bin" / "octave-cli-10.3.0"
    image_dir = runtime_root / "share" / "octave" / "10.3.0" / "m" / "image"
    binary.parent.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    _make_executable(binary)

    startup_env = octave_startup_env(binary)

    assert startup_env["OCTAVE_HOME"] == str(runtime_root)
    assert startup_env["OCTAVE_EXEC_HOME"] == str(runtime_root)
    assert startup_env["OCTAVE_IMAGE_PATH"] == str(image_dir)


def test_run_octave_injects_startup_env(monkeypatch, tmp_path: Path) -> None:
    runtime_root = tmp_path / "env"
    binary = runtime_root / "bin" / "octave-cli-10.3.0"
    image_dir = runtime_root / "share" / "octave" / "10.3.0" / "m" / "image"
    binary.parent.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    _make_executable(binary)

    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("pyisetcam.octave_runner.subprocess.run", fake_run)

    completed = run_octave(["--version"], cwd=tmp_path, octave_bin=binary)

    assert completed.stdout == "ok"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OCTAVE_HOME"] == str(runtime_root)
    assert env["OCTAVE_EXEC_HOME"] == str(runtime_root)
    assert env["OCTAVE_IMAGE_PATH"] == str(image_dir)
