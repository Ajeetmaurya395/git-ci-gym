from __future__ import annotations

import base64
import csv
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import tomllib


ROOT = Path(__file__).resolve().parent


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]


def _dist_name(project_name: str) -> str:
    return project_name.replace("-", "_")


def _dist_info_dir(metadata: dict) -> str:
    return f"{_dist_name(metadata['name'])}-{metadata['version']}.dist-info"


def _wheel_filename(metadata: dict) -> str:
    return f"{_dist_name(metadata['name'])}-{metadata['version']}-py3-none-any.whl"


def _metadata_contents(metadata: dict) -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {metadata['name']}",
        f"Version: {metadata['version']}",
        f"Summary: {metadata.get('description', '')}",
        f"Requires-Python: {metadata.get('requires-python', '>=3.10')}",
    ]

    for requirement in metadata.get("dependencies", []):
        lines.append(f"Requires-Dist: {requirement}")

    for extra_name, extra_requirements in metadata.get(
        "optional-dependencies", {}
    ).items():
        lines.append(f"Provides-Extra: {extra_name}")
        for requirement in extra_requirements:
            lines.append(f"Requires-Dist: {requirement}; extra == '{extra_name}'")

    lines.append("")
    return "\n".join(lines)


def _wheel_contents() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: git_ci_gym.build_backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _record_line(path: str, data: bytes) -> tuple[str, str, str]:
    digest = hashlib.sha256(data).digest()
    encoded_digest = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return path, f"sha256={encoded_digest}", str(len(data))


def _write_wheel(
    wheel_directory: str,
    entries: list[tuple[str, bytes]],
    metadata_directory: str | None = None,
) -> str:
    metadata = _project_metadata()
    dist_info_dir = _dist_info_dir(metadata)
    wheel_name = _wheel_filename(metadata)
    wheel_path = Path(wheel_directory) / wheel_name

    record_rows: list[tuple[str, str, str]] = []
    metadata_bytes = _metadata_contents(metadata).encode()
    wheel_bytes = _wheel_contents().encode()

    metadata_dir = None
    if metadata_directory:
        metadata_dir = Path(metadata_directory) / dist_info_dir
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "METADATA").write_bytes(metadata_bytes)
        (metadata_dir / "WHEEL").write_bytes(wheel_bytes)

    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as wheel_file:
        for archive_path, data in entries:
            wheel_file.writestr(archive_path, data)
            record_rows.append(_record_line(archive_path, data))

        for archive_path, data in (
            (f"{dist_info_dir}/METADATA", metadata_bytes),
            (f"{dist_info_dir}/WHEEL", wheel_bytes),
        ):
            wheel_file.writestr(archive_path, data)
            record_rows.append(_record_line(archive_path, data))

        record_path = f"{dist_info_dir}/RECORD"
        record_buffer = []
        for row in record_rows:
            record_buffer.append(row)
        record_buffer.append((record_path, "", ""))

        csv_output = []
        for row in record_buffer:
            csv_output.append(",".join(row))
        record_bytes = ("\n".join(csv_output) + "\n").encode()
        wheel_file.writestr(record_path, record_bytes)

    return wheel_name


def _editable_entries() -> list[tuple[str, bytes]]:
    return [
        (
            f"{_dist_name(_project_metadata()['name'])}.pth",
            f"{ROOT.parent}{os.linesep}".encode(),
        )
    ]


def _package_entries() -> list[tuple[str, bytes]]:
    files = [
        ("git_ci_gym/__init__.py", ROOT / "__init__.py"),
        ("git_ci_gym/client.py", ROOT / "client.py"),
        ("git_ci_gym/inference.py", ROOT / "inference.py"),
        ("git_ci_gym/models.py", ROOT / "models.py"),
        ("git_ci_gym/openenv.yaml", ROOT / "openenv.yaml"),
        ("server/__init__.py", ROOT / "server" / "__init__.py"),
        ("server/app.py", ROOT / "server" / "app.py"),
        ("server/git_ci_environment.py", ROOT / "server" / "git_ci_environment.py"),
        ("server/tasks.py", ROOT / "server" / "tasks.py"),
    ]
    return [(archive_path, source_path.read_bytes()) for archive_path, source_path in files]


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_editable(config_settings=None) -> list[str]:
    return []


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings=None, metadata_directory_name=None
) -> str:
    metadata = _project_metadata()
    dist_info_dir = _dist_info_dir(metadata)
    target_dir = Path(metadata_directory) / dist_info_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "METADATA").write_text(_metadata_contents(metadata))
    (target_dir / "WHEEL").write_text(_wheel_contents())
    return dist_info_dir


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings=None, metadata_directory_name=None
) -> str:
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(
    wheel_directory: str,
    config_settings=None,
    metadata_directory: str | None = None,
) -> str:
    return _write_wheel(wheel_directory, _package_entries(), metadata_directory)


def build_editable(
    wheel_directory: str,
    config_settings=None,
    metadata_directory: str | None = None,
) -> str:
    return _write_wheel(wheel_directory, _editable_entries(), metadata_directory)


def build_sdist(sdist_directory: str, config_settings=None) -> str:
    metadata = _project_metadata()
    archive_name = f"{_dist_name(metadata['name'])}-{metadata['version']}.tar.gz"
    archive_path = Path(sdist_directory) / archive_name

    with TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir) / f"{_dist_name(metadata['name'])}-{metadata['version']}"
        temp_root.mkdir(parents=True, exist_ok=True)

        for source in [
            ROOT / "__init__.py",
            ROOT / "client.py",
            ROOT / "inference.py",
            ROOT / "models.py",
            ROOT / "openenv.yaml",
            ROOT / "pyproject.toml",
            ROOT / "README.md",
            ROOT / "server",
        ]:
            target = temp_root / source.name
            if source.is_dir():
                import shutil

                shutil.copytree(source, target)
            else:
                target.write_bytes(source.read_bytes())

        import tarfile

        with tarfile.open(archive_path, "w:gz") as tar_file:
            tar_file.add(temp_root, arcname=temp_root.name)

    return archive_name


def _supported_features() -> list[str]:
    return ["build_editable"]
