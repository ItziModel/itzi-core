from __future__ import annotations

import argparse
import re
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

PROJECT_NAME = "itzi-core"
EXPECTED_WHEEL_COUNT = 14


def version_from_tag(tag: str) -> str:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise SystemExit(f"Release tags must use the vX.Y.Z format, got {tag!r}.")
    return tag[1:]


def verify_version(tag: str, pyproject_path: Path) -> None:
    version = version_from_tag(tag)
    project_version = tomllib.loads(pyproject_path.read_text())["project"]["version"]
    if project_version != version:
        raise SystemExit(f"Tag {tag!r} does not match project.version {project_version!r}.")


def verify_artifacts(dist: Path, tag: str) -> None:
    version = version_from_tag(tag)
    sdists = list(dist.glob("*.tar.gz"))
    wheels = list(dist.glob("*.whl"))

    if len(sdists) != 1:
        raise SystemExit(f"Expected one sdist, found {len(sdists)}: {sdists}")
    if len(wheels) != EXPECTED_WHEEL_COUNT:
        raise SystemExit(
            "Expected ten wheels (CPython 3.12 and 3.13 on each of seven platforms), "
            f"found {len(wheels)}: {wheels}"
        )

    with tarfile.open(sdists[0]) as archive:
        pyproject = next(
            (member for member in archive.getmembers() if member.name.endswith("/pyproject.toml")),
            None,
        )
        if pyproject is None:
            raise SystemExit(f"Could not identify pyproject.toml in {sdists[0]}.")
        pyproject_file = archive.extractfile(pyproject)
        if pyproject_file is None:
            raise SystemExit(f"Could not read pyproject.toml from {sdists[0]}.")
        source_version = tomllib.loads(pyproject_file.read().decode())["project"]["version"]
    if source_version != version:
        raise SystemExit(f"Sdist version {source_version!r} does not match {version!r}.")

    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            metadata_files = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise SystemExit(f"Could not identify wheel metadata in {wheel}.")
            metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        if metadata["Name"] != PROJECT_NAME or metadata["Version"] != version:
            raise SystemExit(
                f"Wheel {wheel} has {metadata['Name']} {metadata['Version']}, "
                f"expected {PROJECT_NAME} {version}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate release versions and artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("verify-version")
    version_parser.add_argument("tag")
    version_parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))

    artifacts_parser = subparsers.add_parser("verify-artifacts")
    artifacts_parser.add_argument("dist", type=Path)
    artifacts_parser.add_argument("tag")

    args = parser.parse_args()
    if args.command == "verify-version":
        verify_version(args.tag, args.pyproject)
    else:
        verify_artifacts(args.dist, args.tag)


if __name__ == "__main__":
    main()
