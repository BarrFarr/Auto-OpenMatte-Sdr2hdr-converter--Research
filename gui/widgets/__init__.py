"""
GUI widget panels - the visual components of the application.

Each panel is a self-contained Qt widget that observes AppState for data
and emits signals/calls controllers for user interactions.
"""
from gui.widgets.source_panel import SourcePanel
from gui.widgets.sync_panel import SyncPanel
from gui.widgets.preview_panel import PreviewPanel
from gui.widgets.shot_lock_panel import ShotLockPanel
from gui.widgets.output_preview_panel import OutputPreviewPanel
from gui.widgets.quality_panel import QualityPanel
from gui.widgets.render_panel import RenderPanel

__all__ = [
    "SourcePanel",
    "SyncPanel",
    "PreviewPanel",
    "ShotLockPanel",
    "OutputPreviewPanel",
    "QualityPanel",
    "RenderPanel",
]
