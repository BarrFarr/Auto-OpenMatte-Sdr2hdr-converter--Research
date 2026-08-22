"""
Sync Controller - Manages sync proposals and the lock workflow.

Orchestrates the flow:
1. User triggers auto-sync -> backend computes offset
2. Result is presented as PROPOSAL (not final)
3. User reviews N-1/N/N+1 and adjusts if needed
4. User confirms via LOCK -> shot offset becomes authoritative
5. Fitting can only begin after lock confirmation

This controller does NOT implement sync algorithms.
It calls the backend and manages GUI state transitions.
"""
from PySide6.QtCore import QObject, Signal, Slot

from gui.models.state import AppState
from gui.models.sync_state import (
    SyncProposal,
    SyncQuality,
    ShotLock,
    LockStatus,
)
from gui.controllers.pipeline_adapter import PipelineAdapter


class SyncController(QObject):
    """Manages sync proposal lifecycle and lock workflow."""

    sync_started = Signal()
    sync_finished = Signal()
    lock_confirmed = Signal(str)  # shot_id

    def __init__(
        self, app_state: AppState, adapter: PipelineAdapter, parent=None
    ):
        super().__init__(parent)
        self.app_state = app_state
        self.adapter = adapter

        # Connect adapter signals
        self.adapter.sync_completed.connect(self._on_sync_completed)
        self.adapter.error_occurred.connect(self._on_error)

    def run_auto_sync(self):
        """Initiate auto-sync via the backend.

        Requires both HDR and OM sources to be loaded.
        """
        hdr = self.app_state.hdr_source
        om = self.app_state.om_source

        if not hdr or not om:
            self.app_state.status_message.emit(
                "Cannot sync: both HDR and OM sources required"
            )
            return

        if not hdr.path or not om.path:
            self.app_state.status_message.emit(
                "Cannot sync: source paths not set"
            )
            return

        self.sync_started.emit()
        self.app_state.status_message.emit("Running auto-sync...")
        self.adapter.run_auto_sync(hdr.path, om.path)

    def adjust_offset(self, delta: int):
        """Adjust the current proposal offset by delta frames.

        Args:
            delta: Frame adjustment (positive or negative).
        """
        proposal = self.app_state.sync_proposal
        if not proposal:
            # Create a new proposal if none exists
            proposal = SyncProposal(offset=0)

        proposal.adjust_offset(delta)
        self.app_state.set_sync_proposal(proposal)

    def set_offset(self, offset: int):
        """Set an explicit offset value (manual entry).

        Args:
            offset: The frame offset to set.
        """
        proposal = self.app_state.sync_proposal
        if not proposal:
            proposal = SyncProposal(offset=offset)
        else:
            proposal.set_offset(offset)

        self.app_state.set_sync_proposal(proposal)

    def lock_current_offset(self):
        """Lock the current offset for the active shot.

        Creates a ShotLock from the current proposal and adds it
        to the application state.
        """
        proposal = self.app_state.sync_proposal
        if not proposal:
            self.app_state.status_message.emit(
                "No sync proposal to lock"
            )
            return

        # Mark proposal as accepted
        proposal.is_accepted = True
        self.app_state.set_sync_proposal(proposal)

        # Create shot lock
        shot_index = self.app_state.current_shot_index
        shot_id = f"shot_{shot_index:04d}"

        shot_lock = ShotLock(
            shot_id=shot_id,
            offset=proposal.offset,
            confidence=proposal.confidence,
            lock_status=LockStatus.LOCKED,
        )

        self.app_state.lock_shot(shot_lock)
        self.app_state.status_message.emit(
            f"Locked {shot_id} at offset {proposal.offset}"
        )
        self.lock_confirmed.emit(shot_id)

    def unlock_shot(self, shot_id: str):
        """Unlock a previously locked shot.

        Args:
            shot_id: ID of the shot to unlock.
        """
        for sl in self.app_state.shot_locks:
            if sl.shot_id == shot_id:
                sl.unlock()
                self.app_state.shots_changed.emit()
                self.app_state.status_message.emit(
                    f"Unlocked {shot_id}"
                )
                return

    @Slot(object)
    def _on_sync_completed(self, result):
        """Handle auto-sync completion from backend.

        Args:
            result: Dict with offset/confidence/status, or None on failure.
        """
        self.sync_finished.emit()

        if result is None:
            self.app_state.status_message.emit(
                "Auto-sync failed - no result"
            )
            return

        # Determine quality assessment from backend status
        status_value = result.get("status", "")
        confidence = result.get("confidence", 0.0)

        if confidence >= 0.8 and status_value in ("LOCKED", "locked"):
            quality = SyncQuality.STABLE
        elif confidence >= 0.5:
            quality = SyncQuality.UNCERTAIN
        else:
            quality = SyncQuality.AMBIGUOUS

        # Create proposal (NOT a lock - user must confirm)
        proposal = SyncProposal(
            offset=result.get("offset", 0),
            score=result.get("score", 0.0),
            confidence=confidence,
            quality=quality,
            is_accepted=False,
        )
        # Ensure N-1/N+1 are computed
        proposal.__post_init__()

        self.app_state.set_sync_proposal(proposal)
        self.app_state.status_message.emit(
            f"Auto-sync proposal: offset={proposal.offset}, "
            f"confidence={proposal.confidence:.3f}, "
            f"quality={proposal.quality.value}"
        )

    @Slot(str)
    def _on_error(self, message: str):
        """Handle errors from the adapter."""
        self.sync_finished.emit()
        self.app_state.status_message.emit(f"Sync error: {message}")
