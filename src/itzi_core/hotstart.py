"""
Copyright (C) 2026 Laurent G. Courty

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 2.1
of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from itzi_core.hotstart_models import (
    HotstartMetadataV2,
    HotstartResumeConfig,
    HotstartSimulationState,
)
from itzi_core.itzi_error import HotstartError

if TYPE_CHECKING:
    from itzi_core.domain_data import DomainData


HOTSTART_VERSION = 2
METADATA_FILENAME = "metadata.json"
RASTER_STATE_FILENAME = "raster_state.npz"
SWMM_HOTSTART_FILENAME = "swmm_hotstart.hsf"


def create_hotstart_archive(
    domain_data: DomainData,
    resume_config: HotstartResumeConfig,
    simulation_state: HotstartSimulationState,
    raster_state_bytes: bytes,
    swmm_hotstart_bytes: bytes | None = None,
) -> io.BytesIO:
    """
    Create a hotstart archive from provided state payloads.

    Hashes for ``raster_state_bytes`` and (optionally) ``swmm_hotstart_bytes``
    are computed here and injected into a copy of ``simulation_state`` before
    the archive is serialised.
    """
    # Compute hashes from the binary payloads and inject them into state.
    raster_hash = hashlib.blake2b(raster_state_bytes).hexdigest()
    swmm_hash = None
    if swmm_hotstart_bytes is not None:
        swmm_hash = hashlib.blake2b(swmm_hotstart_bytes).hexdigest()
    simulation_state = simulation_state.model_copy(
        update={"raster_domain_hash": raster_hash, "swmm_hotstart_hash": swmm_hash}
    )

    metadata = HotstartMetadataV2(
        creation_date=datetime.now(UTC),
        itzi_version=version("itzi-core"),
        hotstart_version=HOTSTART_VERSION,
        domain_data=domain_data,
        resume_config=resume_config,
        simulation_state=simulation_state,
    )

    # Create zip archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=8
    ) as zip_file:
        zip_file.writestr(RASTER_STATE_FILENAME, raster_state_bytes)

        if swmm_hotstart_bytes is not None:
            zip_file.writestr(SWMM_HOTSTART_FILENAME, swmm_hotstart_bytes)

        json_str = metadata.model_dump_json(indent=2)
        zip_file.writestr(METADATA_FILENAME, json_str)

    # Reset position for reading
    zip_buffer.seek(0)
    return zip_buffer


class HotstartLoader:
    """Load and validate hotstart archives."""

    def __init__(
        self,
        metadata: HotstartMetadataV2,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes | None,
    ):
        """
        Initialize a validated hotstart loader.
        """
        self._metadata = metadata
        self.raster_state_bytes = raster_state_bytes
        self.swmm_hotstart_bytes = swmm_hotstart_bytes

    @classmethod
    def from_file(cls, path: str | Path) -> HotstartLoader:
        """
        Raises:
            HotstartError: If the archive is invalid or corrupted
        """
        path = Path(path)
        if not path.exists():
            raise HotstartError(f"Hotstart file not found: {path}")

        with open(path, "rb") as f:
            data = f.read()

        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: io.BytesIO | bytes) -> HotstartLoader:
        """
        Raises:
            HotstartError: If the archive is invalid or corrupted
        """
        if isinstance(data, io.BytesIO):
            data.seek(0)
            archive_bytes = data.read()
        else:
            archive_bytes = data

        try:
            zip_buffer = io.BytesIO(archive_bytes)
            zip_file = zipfile.ZipFile(zip_buffer, mode="r")
        except zipfile.BadZipFile as e:
            raise HotstartError("Invalid hotstart archive: not a valid ZIP file") from e

        cls._validate_archive_structure(zip_file)

        # Reject unsupported versions before validating metadata.
        try:
            metadata_bytes = zip_file.read(METADATA_FILENAME)
            metadata_data = json.loads(metadata_bytes)
            archive_version = metadata_data["hotstart_version"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise HotstartError(f"Failed to read hotstart version: {e}") from e

        if type(archive_version) is not int:
            raise HotstartError("Hotstart version must be an integer")

        if archive_version != HOTSTART_VERSION:
            raise HotstartError(
                f"Unsupported hotstart version {archive_version}; "
                f"this build reads only version {HOTSTART_VERSION} archives."
            )

        try:
            metadata = HotstartMetadataV2.model_validate(metadata_data)
        except Exception as e:
            raise HotstartError(f"Failed to validate metadata: {e}") from e

        try:
            raster_state_bytes = zip_file.read(RASTER_STATE_FILENAME)
        except KeyError as e:
            raise HotstartError("Missing raster state in archive") from e

        swmm_hotstart_bytes = None
        if SWMM_HOTSTART_FILENAME in zip_file.namelist():
            swmm_hotstart_bytes = zip_file.read(SWMM_HOTSTART_FILENAME)

        zip_file.close()

        cls._validate_hashes(metadata.simulation_state, raster_state_bytes, swmm_hotstart_bytes)

        return cls(metadata, raster_state_bytes, swmm_hotstart_bytes)

    @staticmethod
    def _validate_archive_structure(zip_file: zipfile.ZipFile) -> None:
        """
        Validate that required archive members are present.

        Raises:
            HotstartError: If required members are missing
        """
        members = set(zip_file.namelist())

        # Check required members
        required = {METADATA_FILENAME, RASTER_STATE_FILENAME}
        missing = required - members

        if missing:
            raise HotstartError(f"Hotstart archive missing required members: {', '.join(missing)}")

    @staticmethod
    def _validate_hashes(
        simulation_state: HotstartSimulationState,
        raster_state_bytes: bytes,
        swmm_hotstart_bytes: bytes | None,
    ) -> None:
        """
        Validate stored BLAKE2 hashes against actual data.

        Raises:
            HotstartError: If hashes don't match or are missing
        """
        # Validate raster state hash
        stored_raster_hash = simulation_state.raster_domain_hash
        computed_raster_hash = hashlib.blake2b(raster_state_bytes).hexdigest()
        if computed_raster_hash != stored_raster_hash:
            raise HotstartError(
                f"Raster state hash mismatch. Archive may be corrupted. "
                f"Expected {stored_raster_hash}, got {computed_raster_hash}"
            )

        stored_swmm_hash = simulation_state.swmm_hotstart_hash

        if swmm_hotstart_bytes is not None:
            if stored_swmm_hash is None:
                raise HotstartError(
                    "Archive contains SWMM hotstart but metadata is missing swmm_hotstart_hash"
                )

            computed_swmm_hash = hashlib.blake2b(swmm_hotstart_bytes).hexdigest()
            if computed_swmm_hash != stored_swmm_hash:
                raise HotstartError(
                    f"SWMM hotstart hash mismatch. Archive may be corrupted. "
                    f"Expected {stored_swmm_hash}, got {computed_swmm_hash}"
                )
        elif stored_swmm_hash is not None:
            raise HotstartError(
                "Metadata indicates SWMM hotstart should be present but it's missing from archive"
            )

    def get_domain_data(self) -> DomainData:
        return self._metadata.domain_data

    def get_resume_config(self) -> HotstartResumeConfig:
        return self._metadata.resume_config

    def get_simulation_state(self) -> HotstartSimulationState:
        return self._metadata.simulation_state

    def get_raster_state_buffer(self) -> io.BytesIO:
        return io.BytesIO(self.raster_state_bytes)

    def get_swmm_hotstart_bytes(self) -> bytes | None:
        return self.swmm_hotstart_bytes

    def has_swmm_hotstart(self) -> bool:
        return self.swmm_hotstart_bytes is not None
