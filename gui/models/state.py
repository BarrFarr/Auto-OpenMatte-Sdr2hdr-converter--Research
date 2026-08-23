"""
Central application state for the GUI.

AppState is the single source of truth for the GUI. It holds references to
source files, sync proposals, shot locks, render status, and the current
project file path. All panels observe AppState via Qt signals.
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from auto_openmatte.core.mode import ProcessingMode
from gui.models.sync_state import SyncProposal, ShotLock


@dataclass
class AudioTrackInfo:
    """Metadata for one selectable audio stream."""

    ordinal: int = 0
    global_index: int = 0
    codec_name: str = ""
    codec_long_name: str = ""
    channels: int = 0
    channel_layout: str = ""
    sample_rate: int = 0
    language: str = ""
    title: str = ""
    is_default: bool = False

    @property
    def label(self) -> str:
        parts = [self.codec_name or "audio"]
        if self.language:
            parts.append(self.language)
        if self.title:
            parts.append(self.title)
        if self.channels:
            parts.append(f"{self.channels}ch")
        return " / ".join(parts)


@dataclass
class SourceFileInfo:
    """Information about a loaded source file (from ffprobe/inspect_source)."""

    path: str = ""
    filename: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_seconds: float = 0.0
    codec: str = ""
    pix_fmt: str = ""
    color_space: str = ""
    color_transfer: str = ""
    color_primaries: str = ""
    hdr_metadata: Optional[dict] = None
    audio_tracks: list[AudioTrackInfo] = field(default_factory=list)
    is_valid: bool = False
    validation_error: str = ""


@dataclass
class RenderStatus:
    """Current render progress information."""

    is_rendering: bool = False
    progress_percent: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    fps: float = 0.0
    eta_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    vram_usage_mb: float = 0.0
    output_path: str = ""
    processing_mode: str = ProcessingMode.EXTEND.value
    error_message: str = ""
    state: str = "IDLE"
    current_segment: int | None = None
    total_segments: int = 0
    last_committed_frame: int = -1
    checkpoint_path: str = ""


@dataclass
class QualityMetrics:
    """Diagnostic quality metrics for the current frame/shot."""

    seam_chroma: Optional[float] = None
    hue_shift: Optional[float] = None
    delta_e_itp: Optional[float] = None
    nonfinite_count: int = 0
    out_of_range_count: int = 0
    frame_index: int = 0


@dataclass
class RenderConfig:
    """Output render configuration."""

    output_path: str = ""
    codec: str = "libx265"
    crf: int = 16
    pix_fmt: str = "yuv420p10le"
    resolution: str = ""  # empty = source resolution
    backend_project_path: str = ""  # canonical analyzed backend project.json
    segment_frames: int = 120
    checkpoint_path: str = ""
    audio_source: str = "NONE"  # NONE, HDR, or OM
    audio_stream_ordinal: int = -1
    audio_stream_index: int = -1
    audio_codec: str = ""
    audio_language: str = ""
    audio_title: str = ""


class AppState(QObject):
    """Central application state with Qt signal notifications.

    All panels connect to signals emitted here to stay synchronized.
    """

    # Signals for state changes
    project_changed = Signal()
    sources_changed = Signal()
    sync_changed = Signal()
    shots_changed = Signal()
    render_status_changed = Signal()
    quality_changed = Signal()
    preview_frame_changed = Signal()
    processing_mode_changed = Signal()
    preview_quality_changed = Signal()
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Project
        self.project_path: Optional[str] = None
        self.project_version: str = "1.0"
        self.is_dirty: bool = False

        # Project processing mode (independent from sync state)
        self.processing_mode: ProcessingMode = ProcessingMode.EXTEND
        self.preview_quality: str = "DRAFT"

        # Sources
        self.hdr_source: Optional[SourceFileInfo] = None
        self.om_source: Optional[SourceFileInfo] = None

        # Sync
        self.sync_proposal: Optional[SyncProposal] = None
        self.shot_locks: list = []  # List[ShotLock]

        # Preview
        self.current_frame: int = 0
        self.current_shot_index: int = 0

        # Render
        self.render_config: RenderConfig = RenderConfig()
        self.render_status: RenderStatus = RenderStatus()

        # Quality
        self.quality_metrics: Optional[QualityMetrics] = None

    def new_project(self):
        """Reset all state to create a new project."""
        self.project_path = None
        self.is_dirty = False
        self.hdr_source = None
        self.om_source = None
        self.sync_proposal = None
        self.shot_locks = []
        self.processing_mode = ProcessingMode.EXTEND
        self.preview_quality = "DRAFT"
        self.current_frame = 0
        self.current_shot_index = 0
        self.render_config = RenderConfig()
        self.render_status = RenderStatus()
        self.quality_metrics = None
        self.project_changed.emit()
        self.sources_changed.emit()
        self.sync_changed.emit()
        self.shots_changed.emit()
        self.processing_mode_changed.emit()
        self.preview_quality_changed.emit()

    def load_project(self, path: str) -> bool:
        """Load project from .omhdr file.

        Args:
            path: Path to the .omhdr project file.

        Returns:
            True if loaded successfully.
        """
        from gui.models.project import ProjectFile

        try:
            project_file = ProjectFile.load(path)
            self._apply_project_data(project_file)
            self.project_path = path
            self.is_dirty = False
            self.project_changed.emit()
            self.sources_changed.emit()
            self.sync_changed.emit()
            self.shots_changed.emit()
            self.processing_mode_changed.emit()
            return True
        except Exception as e:
            self.status_message.emit(f"Load failed: {e}")
            return False

    def save_project(self, path: Optional[str] = None) -> bool:
        """Save current state to .omhdr file.

        Args:
            path: Save path. If None, uses current project_path.

        Returns:
            True if saved successfully.
        """
        from gui.models.project import ProjectFile

        save_path = path or self.project_path
        if not save_path:
            return False

        try:
            project_file = self._build_project_data()
            project_file.save(save_path)
            self.project_path = save_path
            self.is_dirty = False
            self.project_changed.emit()
            return True
        except Exception as e:
            self.status_message.emit(f"Save failed: {e}")
            return False

    def set_hdr_source(self, info: SourceFileInfo):
        """Update HDR source file info."""
        self.hdr_source = info
        self.is_dirty = True
        self.sources_changed.emit()
        self.project_changed.emit()

    def set_om_source(self, info: SourceFileInfo):
        """Update OpenMatte source file info."""
        self.om_source = info
        self.is_dirty = True
        self.sources_changed.emit()
        self.project_changed.emit()

    def set_sync_proposal(self, proposal: SyncProposal):
        """Update sync proposal from auto-sync."""
        self.sync_proposal = proposal
        self.is_dirty = True
        self.sync_changed.emit()
        self.project_changed.emit()

    def lock_shot(self, shot_lock: ShotLock):
        """Add or update a shot lock."""
        # Replace existing lock for same shot_id
        self.shot_locks = [
            sl for sl in self.shot_locks if sl.shot_id != shot_lock.shot_id
        ]
        self.shot_locks.append(shot_lock)
        self.is_dirty = True
        self.shots_changed.emit()
        self.project_changed.emit()

    def set_processing_mode(self, mode: ProcessingMode | str):
        """Set the project-level apply mode without touching synchronization."""
        normalized = ProcessingMode.coerce(mode)
        if normalized == self.processing_mode:
            return
        self.processing_mode = normalized
        self.is_dirty = True
        self.processing_mode_changed.emit()
        self.project_changed.emit()

    def set_preview_quality(self, quality: str):
        """Set the explicit preview quality badge (DRAFT or FULL)."""
        normalized = str(quality).upper()
        if normalized not in {"DRAFT", "FULL"}:
            raise ValueError("Preview quality must be DRAFT or FULL")
        if normalized == self.preview_quality:
            return
        self.preview_quality = normalized
        self.preview_quality_changed.emit()

    def set_current_frame(self, frame: int):
        """Update current preview frame."""
        if frame < 0:
            frame = 0
        self.current_frame = frame
        self.preview_frame_changed.emit()

    def set_render_status(self, status: RenderStatus):
        """Update render progress status."""
        self.render_status = status
        self.render_status_changed.emit()

    def set_quality_metrics(self, metrics: QualityMetrics):
        """Update quality diagnostic metrics."""
        self.quality_metrics = metrics
        self.quality_changed.emit()

    def _apply_project_data(self, project_file):
        """Apply loaded project data to state."""
        self.project_version = project_file.version
        self.processing_mode = ProcessingMode.coerce(project_file.processing_mode)

        if project_file.hdr_source_path:
            self.hdr_source = SourceFileInfo(
                path=project_file.hdr_source_path,
                filename=Path(project_file.hdr_source_path).name,
                is_valid=True,
            )
        if project_file.om_source_path:
            self.om_source = SourceFileInfo(
                path=project_file.om_source_path,
                filename=Path(project_file.om_source_path).name,
                is_valid=True,
            )

        self.shot_locks = project_file.shot_locks
        self.render_config = project_file.render_config or RenderConfig()

    def _build_project_data(self):
        """Build ProjectFile from current state."""
        from gui.models.project import ProjectFile

        return ProjectFile(
            version=self.project_version,
            processing_mode=self.processing_mode.value,
            hdr_source_path=self.hdr_source.path if self.hdr_source else "",
            om_source_path=self.om_source.path if self.om_source else "",
            shot_locks=list(self.shot_locks),
            render_config=self.render_config,
        )
