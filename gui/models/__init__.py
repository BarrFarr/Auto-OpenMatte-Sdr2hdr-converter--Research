"""
GUI application models - state management, project persistence, sync state.

These models are GUI-level representations. They reference but do not modify
the backend data models in auto_openmatte.core.models.
"""
from gui.models.state import AppState
from gui.models.project import ProjectFile
from gui.models.sync_state import SyncProposal, ShotLock

__all__ = ["AppState", "ProjectFile", "SyncProposal", "ShotLock"]
