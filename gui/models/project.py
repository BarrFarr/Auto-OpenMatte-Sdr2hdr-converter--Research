"""
Project file (.omhdr) save/load functionality.

The .omhdr format is a JSON file that stores:
- Project version
- Source file paths (HDR and OpenMatte)
- Shot offsets and lock status
- Render parameters
- Sync proposal state

This wraps the patterns from auto_openmatte.core.project but adds
GUI-specific state that the backend does not track.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from gui.models.sync_state import ShotLock, SyncProposal


@dataclass
class ProjectFile:
    """Represents the .omhdr project file format.

    Stores all persistent GUI state needed to resume a session.
    """

    version: str = "1.0"
    hdr_source_path: str = ""
    om_source_path: str = ""
    shot_locks: list = field(default_factory=list)  # List[ShotLock]
    render_config: Optional[object] = None
    sync_proposal: Optional[SyncProposal] = None

    def save(self, path: str):
        """Save project to .omhdr JSON file.

        Args:
            path: File path to save to (should end with .omhdr)
        """
        data = {
            "format": "omhdr",
            "version": self.version,
            "sources": {
                "hdr_path": self.hdr_source_path,
                "om_path": self.om_source_path,
            },
            "shot_locks": [sl.to_dict() for sl in self.shot_locks],
            "render_config": self._serialize_render_config(),
            "sync_proposal": (
                self.sync_proposal.to_dict()
                if self.sync_proposal
                else None
            ),
        }

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ProjectFile":
        """Load project from .omhdr JSON file.

        Args:
            path: Path to the .omhdr file.

        Returns:
            Populated ProjectFile instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Project file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("format") != "omhdr":
            raise ValueError(
                f"Invalid project format: {data.get('format', 'unknown')}"
            )

        sources = data.get("sources", {})
        shot_locks_data = data.get("shot_locks", [])
        shot_locks = [ShotLock.from_dict(sl) for sl in shot_locks_data]

        sync_data = data.get("sync_proposal")
        sync_proposal = (
            SyncProposal.from_dict(sync_data) if sync_data else None
        )

        render_config = cls._deserialize_render_config(
            data.get("render_config")
        )

        return cls(
            version=data.get("version", "1.0"),
            hdr_source_path=sources.get("hdr_path", ""),
            om_source_path=sources.get("om_path", ""),
            shot_locks=shot_locks,
            render_config=render_config,
            sync_proposal=sync_proposal,
        )

    def _serialize_render_config(self) -> Optional[dict]:
        """Serialize render config to dict."""
        if self.render_config is None:
            return None
        # RenderConfig is a dataclass from state.py
        rc = self.render_config
        return {
            "output_path": getattr(rc, "output_path", ""),
            "codec": getattr(rc, "codec", "libx265"),
            "crf": getattr(rc, "crf", 16),
            "pix_fmt": getattr(rc, "pix_fmt", "yuv420p10le"),
            "resolution": getattr(rc, "resolution", ""),
        }

    @classmethod
    def _deserialize_render_config(cls, data: Optional[dict]):
        """Deserialize render config from dict."""
        if data is None:
            return None
        # Import here to avoid circular imports
        from gui.models.state import RenderConfig

        return RenderConfig(
            output_path=data.get("output_path", ""),
            codec=data.get("codec", "libx265"),
            crf=data.get("crf", 16),
            pix_fmt=data.get("pix_fmt", "yuv420p10le"),
            resolution=data.get("resolution", ""),
        )
