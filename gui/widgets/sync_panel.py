"""
Sync Panel - Auto-sync proposal display and manual offset controls.

Auto-sync is treated as a PROPOSAL only. The user must explicitly confirm
(LOCK) the offset before it becomes authoritative. The panel shows:
- Auto-sync result: offset N-1, N, N+1 simultaneously
- Quality assessment: stable / uncertain / ambiguous
- Manual adjustment: -1/+1, -10/+10, direct entry
- LOCK button to confirm the shot-level offset

Offset is shot-level, not per-frame.
"""
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QGridLayout,
    QFrame,
)
from PySide6.QtCore import Qt, Slot, Signal

from gui.models.state import AppState
from gui.models.sync_state import SyncProposal, SyncQuality, ShotLock, LockStatus
from gui.controllers.sync_controller import SyncController


class OffsetComparisonWidget(QWidget):
    """Shows offset N-1, N, and N+1 for visual comparison."""

    offset_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # N-1 column
        self.n_minus_1_frame = QFrame(self)
        self.n_minus_1_frame.setFrameShape(QFrame.Shape.Box)
        n_minus_1_layout = QVBoxLayout(self.n_minus_1_frame)
        self.n_minus_1_label = QLabel("N-1", self)
        self.n_minus_1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.n_minus_1_value = QLabel("--", self)
        self.n_minus_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.n_minus_1_value.setProperty("class", "offset-value")
        self.n_minus_1_btn = QPushButton("Use N-1", self)
        n_minus_1_layout.addWidget(self.n_minus_1_label)
        n_minus_1_layout.addWidget(self.n_minus_1_value)
        n_minus_1_layout.addWidget(self.n_minus_1_btn)
        layout.addWidget(self.n_minus_1_frame)

        # N column (proposed / current)
        self.n_frame = QFrame(self)
        self.n_frame.setFrameShape(QFrame.Shape.Box)
        self.n_frame.setProperty("class", "current-offset")
        n_layout = QVBoxLayout(self.n_frame)
        self.n_label = QLabel("N (Proposed)", self)
        self.n_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.n_value = QLabel("--", self)
        self.n_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.n_value.setProperty("class", "offset-value-primary")
        n_layout.addWidget(self.n_label)
        n_layout.addWidget(self.n_value)
        layout.addWidget(self.n_frame)

        # N+1 column
        self.n_plus_1_frame = QFrame(self)
        self.n_plus_1_frame.setFrameShape(QFrame.Shape.Box)
        n_plus_1_layout = QVBoxLayout(self.n_plus_1_frame)
        self.n_plus_1_label = QLabel("N+1", self)
        self.n_plus_1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.n_plus_1_value = QLabel("--", self)
        self.n_plus_1_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.n_plus_1_value.setProperty("class", "offset-value")
        self.n_plus_1_btn = QPushButton("Use N+1", self)
        n_plus_1_layout.addWidget(self.n_plus_1_label)
        n_plus_1_layout.addWidget(self.n_plus_1_value)
        n_plus_1_layout.addWidget(self.n_plus_1_btn)
        layout.addWidget(self.n_plus_1_frame)

        # Connect buttons
        self.n_minus_1_btn.clicked.connect(lambda: self._emit_offset(-1))
        self.n_plus_1_btn.clicked.connect(lambda: self._emit_offset(1))

    def _emit_offset(self, delta: int):
        """Emit signal when user selects an adjacent offset."""
        self.offset_selected.emit(delta)

    def update_offsets(self, proposal: SyncProposal):
        """Update displayed offset values."""
        self.n_minus_1_value.setText(str(proposal.offset_minus_1))
        self.n_value.setText(str(proposal.offset))
        self.n_plus_1_value.setText(str(proposal.offset_plus_1))


class SyncPanel(QWidget):
    """Panel for sync proposal display and manual offset controls."""

    def __init__(
        self, app_state: AppState, sync_controller: SyncController, parent=None
    ):
        super().__init__(parent)
        self.app_state = app_state
        self.sync_controller = sync_controller

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Create the panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Auto-sync section
        auto_group = QGroupBox("Auto-Sync Proposal", self)
        auto_layout = QVBoxLayout(auto_group)

        # Run auto-sync button
        self.auto_sync_btn = QPushButton("Run Auto-Sync", self)
        self.auto_sync_btn.setProperty("class", "primary-button")
        auto_layout.addWidget(self.auto_sync_btn)

        # Status display
        status_layout = QGridLayout()
        status_layout.addWidget(QLabel("Status:"), 0, 0)
        self.quality_label = QLabel("-", self)
        self.quality_label.setProperty("class", "sync-quality")
        status_layout.addWidget(self.quality_label, 0, 1)

        status_layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.confidence_label = QLabel("-", self)
        status_layout.addWidget(self.confidence_label, 1, 1)

        status_layout.addWidget(QLabel("Score:"), 2, 0)
        self.score_label = QLabel("-", self)
        status_layout.addWidget(self.score_label, 2, 1)

        auto_layout.addLayout(status_layout)
        main_layout.addWidget(auto_group)

        # Offset comparison (N-1 / N / N+1)
        comparison_group = QGroupBox("Offset Comparison", self)
        comparison_layout = QVBoxLayout(comparison_group)
        self.offset_comparison = OffsetComparisonWidget(self)
        comparison_layout.addWidget(self.offset_comparison)
        main_layout.addWidget(comparison_group)

        # Manual controls
        manual_group = QGroupBox("Manual Offset Adjustment", self)
        manual_layout = QVBoxLayout(manual_group)

        # Fine adjustment: -1 / +1
        fine_layout = QHBoxLayout()
        self.minus_1_btn = QPushButton("-1", self)
        self.plus_1_btn = QPushButton("+1", self)
        self.minus_1_btn.setFixedWidth(60)
        self.plus_1_btn.setFixedWidth(60)
        fine_layout.addWidget(QLabel("Fine:"))
        fine_layout.addWidget(self.minus_1_btn)
        fine_layout.addWidget(self.plus_1_btn)
        fine_layout.addStretch()
        manual_layout.addLayout(fine_layout)

        # Coarse adjustment: -10 / +10
        coarse_layout = QHBoxLayout()
        self.minus_10_btn = QPushButton("-10", self)
        self.plus_10_btn = QPushButton("+10", self)
        self.minus_10_btn.setFixedWidth(60)
        self.plus_10_btn.setFixedWidth(60)
        coarse_layout.addWidget(QLabel("Coarse:"))
        coarse_layout.addWidget(self.minus_10_btn)
        coarse_layout.addWidget(self.plus_10_btn)
        coarse_layout.addStretch()
        manual_layout.addLayout(coarse_layout)

        # Direct entry
        entry_layout = QHBoxLayout()
        entry_layout.addWidget(QLabel("Set Offset:"))
        self.offset_spinbox = QSpinBox(self)
        self.offset_spinbox.setRange(-10000, 10000)
        self.offset_spinbox.setValue(0)
        entry_layout.addWidget(self.offset_spinbox)
        self.apply_offset_btn = QPushButton("Apply", self)
        entry_layout.addWidget(self.apply_offset_btn)
        entry_layout.addStretch()
        manual_layout.addLayout(entry_layout)

        main_layout.addWidget(manual_group)

        # Lock section
        lock_group = QGroupBox("Confirm Sync", self)
        lock_layout = QVBoxLayout(lock_group)

        self.lock_info_label = QLabel(
            "Lock the current offset to proceed to fitting.\n"
            "Fitting cannot begin until sync is confirmed.",
            self,
        )
        self.lock_info_label.setWordWrap(True)
        lock_layout.addWidget(self.lock_info_label)

        self.lock_btn = QPushButton("LOCK", self)
        self.lock_btn.setProperty("class", "lock-button")
        self.lock_btn.setMinimumHeight(40)
        lock_layout.addWidget(self.lock_btn)

        self.lock_status_label = QLabel("Status: UNLOCKED", self)
        lock_layout.addWidget(self.lock_status_label)

        main_layout.addWidget(lock_group)
        main_layout.addStretch()

    def _connect_signals(self):
        """Connect UI signals."""
        self.auto_sync_btn.clicked.connect(self._run_auto_sync)
        self.minus_1_btn.clicked.connect(lambda: self._adjust_offset(-1))
        self.plus_1_btn.clicked.connect(lambda: self._adjust_offset(1))
        self.minus_10_btn.clicked.connect(lambda: self._adjust_offset(-10))
        self.plus_10_btn.clicked.connect(lambda: self._adjust_offset(10))
        self.apply_offset_btn.clicked.connect(self._apply_manual_offset)
        self.lock_btn.clicked.connect(self._lock_offset)
        self.offset_comparison.offset_selected.connect(self._adjust_offset)

        # State observation
        self.app_state.sync_changed.connect(self._refresh_display)

    @Slot()
    def _run_auto_sync(self):
        """Invoke auto-sync via the controller."""
        self.auto_sync_btn.setEnabled(False)
        self.app_state.status_message.emit("Running auto-sync...")
        self.sync_controller.run_auto_sync()
        self.auto_sync_btn.setEnabled(True)

    @Slot(int)
    def _adjust_offset(self, delta: int):
        """Adjust the proposed offset by delta frames."""
        self.sync_controller.adjust_offset(delta)

    @Slot()
    def _apply_manual_offset(self):
        """Apply manually entered offset value."""
        offset = self.offset_spinbox.value()
        self.sync_controller.set_offset(offset)

    @Slot()
    def _lock_offset(self):
        """Lock the current offset."""
        self.sync_controller.lock_current_offset()

    @Slot()
    def _refresh_display(self):
        """Update display from current state."""
        proposal = self.app_state.sync_proposal
        if not proposal:
            return

        # Update quality indicator
        quality_text = proposal.quality.value.upper()
        self.quality_label.setText(quality_text)

        quality_colors = {
            SyncQuality.STABLE: "color: #44cc44;",
            SyncQuality.UNCERTAIN: "color: #ffaa00;",
            SyncQuality.AMBIGUOUS: "color: #ff4444;",
        }
        self.quality_label.setStyleSheet(
            quality_colors.get(proposal.quality, "")
        )

        # Update confidence and score
        self.confidence_label.setText(f"{proposal.confidence:.3f}")
        self.score_label.setText(f"{proposal.score:.4f}")

        # Update offset comparison
        self.offset_comparison.update_offsets(proposal)
        self.offset_spinbox.setValue(proposal.offset)

        # Update lock status
        if proposal.is_accepted:
            self.lock_status_label.setText("Status: LOCKED")
            self.lock_status_label.setStyleSheet("color: #44cc44;")
            self.lock_btn.setEnabled(False)
        else:
            self.lock_status_label.setText("Status: UNLOCKED")
            self.lock_status_label.setStyleSheet("color: #ffaa00;")
            self.lock_btn.setEnabled(True)
