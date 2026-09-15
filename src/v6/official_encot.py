"""Pin, audit, and locate a pre-existing runtime for official EnCOT.

This module never installs Python packages and never executes EnCOT's ``main.py``
or shell scripts.  Only the exact model-core files at the frozen commit are
loaded later by the isolated Step-3 worker.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _run_bytes(command: list[str], *, cwd: Path | None = None) -> bytes:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _canonical_git_file_sha256(root: Path, relative: str) -> str:
    """Hash committed blob bytes, independent of Windows checkout line endings."""

    blob = _run_bytes(["git", "show", f"HEAD:{relative}"], cwd=root)
    return sha256(blob).hexdigest()


def _normalized_repository(value: str) -> str:
    normalized = value.strip().replace("\\", "/").casefold().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    for prefix in (
        "https://www.github.com/",
        "http://www.github.com/",
        "https://github.com/",
        "http://github.com/",
        "ssh://git@github.com/",
        "git@github.com:",
    ):
        if normalized.startswith(prefix):
            return "github.com/" + normalized[len(prefix) :]
    return normalized


def audit_official_source(
    source: Path, specification: Mapping[str, Any]
) -> dict[str, Any]:
    root = source.expanduser().resolve()
    if not (root / ".git").exists():
        raise ValueError(f"Official EnCOT source is not a git checkout: {root}")
    origin = _run_text(["git", "remote", "get-url", "origin"], cwd=root)
    if _normalized_repository(origin) != _normalized_repository(
        str(specification["repository"])
    ):
        raise ValueError(f"Unexpected EnCOT origin: {origin}")
    head = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    if head != str(specification["commit"]):
        raise ValueError(
            f"EnCOT HEAD {head} is not the locked commit {specification['commit']}"
        )
    tree = _run_text(["git", "rev-parse", "HEAD^{tree}"], cwd=root)
    if tree != str(specification["git_tree"]):
        raise ValueError(f"EnCOT tree {tree} is not the locked tree")
    tracked_status = _run_text(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root
    )
    if tracked_status:
        raise RuntimeError(
            "The candidate official EnCOT checkout has tracked modifications; "
            "V6 will not alter it."
        )
    all_status = _run_text(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root
    )
    untracked = [
        line[3:]
        for line in all_status.splitlines()
        if line.startswith("?? ")
    ]
    observed_files: dict[str, str] = {}
    worktree_files: dict[str, str] = {}
    for relative, expected in specification["required_files_sha256"].items():
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Pinned EnCOT file is missing: {relative}")
        # Git may materialize CRLF bytes on Windows even though the committed
        # blob is LF.  The commit/tree identity and clean tracked status prove
        # that the checked-out source is unmodified; compare the frozen hash
        # against the canonical blob so the audit is platform-independent.
        observed = _canonical_git_file_sha256(root, str(relative))
        if observed != str(expected):
            raise ValueError(f"Pinned EnCOT file hash mismatch: {relative}")
        observed_files[str(relative)] = observed
        worktree_files[str(relative)] = sha256_file(path)
    license_files = sorted(
        path.name
        for pattern in ("LICENSE*", "COPYING*")
        for path in root.glob(pattern)
        if path.is_file()
    )
    return {
        "schema_version": 1,
        "method": "EnCOT",
        "repository": str(specification["repository"]),
        "source_directory": str(root),
        "commit": head,
        "git_tree": tree,
        "source_tree_clean": True,
        "source_tree_clean_definition": "no_tracked_modifications",
        "untracked_files_present": len(untracked),
        "untracked_files_loaded": False,
        "required_file_sha256": observed_files,
        "required_file_hash_basis": "canonical_committed_git_blob_bytes",
        "required_worktree_file_sha256": worktree_files,
        "checkout_line_endings_may_be_platform_native": True,
        "required_files_verified": len(observed_files),
        "official_main_executed": False,
        "official_shell_scripts_executed": False,
        "source_redistributed_in_v6_package": False,
        "license_file_detected_at_repository_root": bool(license_files),
        "license_filenames": license_files,
    }


def _source_candidates(
    *, v6_root: Path, source_root: Path, explicit: Path | None
) -> list[Path]:
    values: list[Path] = []
    if explicit is not None:
        values.append(explicit.expanduser().resolve())
    values.extend(
        [
            source_root / "third_party" / "sota" / "EnCOT",
            source_root / "third_party" / "EnCOT",
            v6_root / "third_party" / "official" / "EnCOT",
            v6_root / "third_party" / "sota" / "EnCOT",
            v6_root.parent / "EnCOT",
        ]
    )
    # Search the other SC-HTM project copies under the same PyCharmProjects
    # directory.  This reuses an already downloaded official checkout from V4
    # or V5 even when it is not located under the current source_root.
    for project in sorted(v6_root.parent.glob("SC-HTM*")):
        values.extend(
            [
                project / "third_party" / "sota" / "EnCOT",
                project / "third_party" / "official" / "EnCOT",
                project / "third_party" / "EnCOT",
                project / "EnCOT",
            ]
        )
        third_party = project / "third_party"
        if third_party.is_dir():
            for marker in sorted(third_party.rglob(".git")):
                values.append(marker.parent)
    return list(dict.fromkeys(path.resolve() for path in values))


def _can_supply_locked_commit(
    source: Path, specification: Mapping[str, Any]
) -> bool:
    """Whether a local repository can seed a new clean locked checkout."""

    root = source.expanduser().resolve()
    if not (root / ".git").exists():
        return False
    try:
        origin = _run_text(["git", "remote", "get-url", "origin"], cwd=root)
        tree = _run_text(
            ["git", "rev-parse", f"{specification['commit']}^{{tree}}"],
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(
        _normalized_repository(origin)
        == _normalized_repository(str(specification["repository"]))
        and tree == str(specification["git_tree"])
    )


def _next_managed_destination(v6_root: Path, commit: str) -> Path:
    parent = v6_root / "third_party" / "official"
    parent.mkdir(parents=True, exist_ok=True)
    names = ["EnCOT", f"EnCOT_{commit[:12]}"]
    names.extend(f"EnCOT_{commit[:12]}_{index:02d}" for index in range(2, 100))
    for name in names:
        destination = parent / name
        if not destination.exists():
            return destination
    raise RuntimeError("No unused managed EnCOT checkout destination remains")


def _clone_failure_text(error: subprocess.CalledProcessError) -> str:
    stderr = (error.stderr or "").strip()
    stdout = (error.stdout or "").strip()
    detail = stderr or stdout or "git returned no diagnostic text"
    return f"exit={error.returncode}: {detail}"


def locate_or_clone_official_source(
    *,
    v6_root: Path,
    source_root: Path,
    explicit: Path | None,
    specification: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    rejected: list[dict[str, str]] = []
    local_seeds: list[Path] = []
    for candidate in _source_candidates(
        v6_root=v6_root, source_root=source_root, explicit=explicit
    ):
        if not candidate.exists():
            continue
        try:
            receipt = audit_official_source(candidate, specification)
            receipt["acquisition"] = "REUSED_EXISTING_READ_ONLY_CHECKOUT"
            receipt["rejected_candidates_before_selection"] = rejected
            return candidate, receipt
        except Exception as error:
            rejected.append(
                {
                    "path": str(candidate),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            if _can_supply_locked_commit(candidate, specification):
                local_seeds.append(candidate)
            if explicit is not None and candidate == explicit.expanduser().resolve():
                raise

    destination = _next_managed_destination(
        v6_root, str(specification["commit"])
    )
    if local_seeds:
        clone_source = local_seeds[0]
        command = [
            "git",
            "clone",
            "--no-tags",
            "--no-hardlinks",
            str(clone_source),
            str(destination),
        ]
        acquisition = "CLONED_FROM_EXISTING_LOCAL_OFFICIAL_REPOSITORY"
    else:
        clone_source = None
        command = [
            "git",
            "clone",
            "--no-tags",
            "--no-checkout",
            "--depth=1",
            "--filter=blob:none",
            str(specification["repository"]),
            str(destination),
        ]
        acquisition = "CLONED_SPARSE_LOCKED_COMMIT_NO_PACKAGE_INSTALL"
    try:
        _run_text(command)
        if clone_source is not None:
            _run_text(
                ["git", "remote", "set-url", "origin", str(specification["repository"])],
                cwd=destination,
            )
        else:
            # The repository contains large example datasets that Step 3 never
            # uses.  Materialize only the official model and audit code needed
            # by the locked bridge; HEAD still retains the complete Git tree.
            _run_text(["git", "sparse-checkout", "init", "--cone"], cwd=destination)
            _run_text(
                [
                    "git",
                    "sparse-checkout",
                    "set",
                    "topmost/models/NewMethod",
                    "topmost/trainers",
                    "topmost/data",
                ],
                cwd=destination,
            )
        try:
            _run_text(
                ["git", "checkout", "--detach", str(specification["commit"])],
                cwd=destination,
            )
        except subprocess.CalledProcessError:
            _run_text(
                [
                    "git",
                    "fetch",
                    "--depth=1",
                    "origin",
                    str(specification["commit"]),
                ],
                cwd=destination,
            )
            _run_text(
                ["git", "checkout", "--detach", str(specification["commit"])],
                cwd=destination,
            )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "No reusable local EnCOT checkout passed the lock and the fallback "
            f"clone failed ({_clone_failure_text(error)}). V6 changed no Python "
            "package or environment. Candidate audit: "
            + json.dumps(rejected, ensure_ascii=False)
        ) from error
    receipt = audit_official_source(destination, specification)
    receipt["acquisition"] = acquisition
    receipt["local_clone_seed"] = str(clone_source) if clone_source else None
    receipt["rejected_candidates_before_selection"] = rejected
    return destination, receipt


def _runtime_probe(python: Path) -> dict[str, Any] | None:
    requested = [
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "sentence-transformers",
        "gensim",
        "tqdm",
        "wandb",
        "topmost",
        "geomloss",
    ]
    code = (
        "import importlib.metadata as m,json,sys,torch,torchvision,numpy,scipy,sklearn,geomloss;"
        f"names={requested!r};"
        "print(json.dumps({'python':list(sys.version_info[:3]),"
        "'executable':sys.executable,'cuda_available':torch.cuda.is_available(),"
        "'cuda_runtime':torch.version.cuda,"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'packages':{n:m.version(n) for n in names}},sort_keys=True))"
    )
    try:
        output = _run_text([str(python), "-c", code])
        return json.loads(output.splitlines()[-1])
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        return None


def _base_version(value: Any) -> str:
    return str(value).split("+", maxsplit=1)[0]


def _runtime_matches(
    probe: Mapping[str, Any] | None, specification: Mapping[str, Any]
) -> bool:
    if not probe:
        return False
    required_python = list(map(int, specification["python_major_minor"]))
    if list(map(int, probe.get("python", [])[:2])) != required_python:
        return False
    if bool(specification["require_cuda"]) and probe.get("cuda_available") is not True:
        return False
    observed = probe.get("packages", {})
    for name, expected in specification["packages"].items():
        value = observed.get(str(name))
        if value is None or _base_version(value) != str(expected):
            return False
    return True


def _runtime_candidates(
    *, v6_root: Path, source_root: Path, explicit: Path | None
) -> Iterable[Path]:
    values: list[Path] = []
    if explicit is not None:
        values.append(explicit.expanduser().resolve())
    for root in (source_root, v6_root):
        for pattern in (
            "cache/sota_envs/*/Scripts/python.exe",
            "cache/sota_envs/*/bin/python",
            ".venvs/*/Scripts/python.exe",
            ".venvs/*/bin/python",
        ):
            values.extend(sorted(root.glob(pattern)))
    values.append(Path(sys.executable).resolve())
    return list(dict.fromkeys(path.resolve() for path in values if path.is_file()))


def locate_exact_runtime(
    *,
    v6_root: Path,
    source_root: Path,
    explicit: Path | None,
    specification: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if specification.get("creation_or_installation_allowed") is not False:
        raise ValueError("Step 3 must explicitly prohibit environment installation")
    inspected: list[dict[str, Any]] = []
    for candidate in _runtime_candidates(
        v6_root=v6_root, source_root=source_root, explicit=explicit
    ):
        probe = _runtime_probe(candidate)
        inspected.append(
            {
                "python": str(candidate),
                "probe_succeeded": probe is not None,
                "matches_exact_lock": _runtime_matches(probe, specification),
                "python_version": probe.get("python") if probe else None,
                "packages": probe.get("packages") if probe else None,
                "cuda_available": probe.get("cuda_available") if probe else None,
            }
        )
        if _runtime_matches(probe, specification):
            return candidate, {
                "schema_version": 1,
                "status": "PASS",
                "selected_python": str(candidate),
                "probe": probe,
                "exact_package_lock_matched": True,
                "environment_created": False,
                "packages_installed_or_changed": False,
                "candidates_inspected": inspected,
            }
    if explicit is not None:
        prefix = f"The explicit EnCOT interpreter {explicit} did not match. "
    else:
        prefix = "No exact pre-existing EnCOT interpreter was found. "
    raise RuntimeError(
        prefix
        + "Step 3 deliberately performed no installation. Expected the V4 cached "
        "legacy runtime (Python 3.10, torch 2.4.1, torchvision 0.19.1, CUDA and "
        "the pinned scientific packages). Inspected: "
        + json.dumps(inspected, ensure_ascii=False)
    )
