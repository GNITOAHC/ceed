"""Content-addressed image storage.

An intervened image and the example it came from must be byte-identical for the
teacher and the Student, or their forward passes diverge for reasons unrelated to
the signal. Addressing images by the hash of their bytes makes that guarantee
structural: the same bytes always have the same address, and reading an address
back always returns those exact bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def image_fingerprint(data: bytes) -> str:
    """Return the content address of image bytes.

    Args:
        data: The raw encoded image bytes.

    Returns:
        The hex SHA-256 of the bytes.
    """
    return hashlib.sha256(data).hexdigest()


class ImageStore:
    """A directory of images addressed by the hash of their bytes."""

    def __init__(self, root: Path) -> None:
        """Open (creating if needed) the store rooted at ``root``."""
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, fingerprint: str) -> Path:
        """Return the on-disk path for a fingerprint (whether or not it exists)."""
        return self.root / fingerprint

    def put(self, data: bytes) -> str:
        """Store image bytes and return their fingerprint.

        Writing is idempotent: storing bytes already present is a no-op, so
        multiple workers can store the same intervened image without conflict.

        Args:
            data: The raw encoded image bytes.

        Returns:
            The content address of the stored bytes.
        """
        fingerprint = image_fingerprint(data)
        path = self.path(fingerprint)
        if not path.exists():
            path.write_bytes(data)
        return fingerprint

    def get(self, fingerprint: str) -> bytes:
        """Return the exact bytes stored under ``fingerprint``."""
        return self.path(fingerprint).read_bytes()

    def contains(self, fingerprint: str) -> bool:
        """Return whether an image with this fingerprint is stored."""
        return self.path(fingerprint).exists()
