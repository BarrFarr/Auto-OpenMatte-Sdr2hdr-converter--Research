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
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from auto_openmatte.core.config import FAST_SYNC_ANALYSIS_MINUTES
from gui.controllers.sync_controller import SyncController
from gui.models.state import AppState
from gui.models.sync_state import SyncProposal, SyncQuality


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

        # Optional bounded fast strategy
        fast_controls_layout = QHBoxLayout()
        self.fast_sync_btn = QPushButton("FAST AUTO SYNC (GPU)", self)
        self.fast_sync_btn.setProperty("class", "secondary-button")
        fast_controls_layout.addWidget(self.fast_sync_btn, 1)
        fast_controls_layout.addWidget(QLabel("Analyze:", self))
        self.fast_analysis_combo = QComboBox(self)
        for minutes in FAST_SYNC_ANALYSIS_MINUTES:
            self.fast_analysis_combo.addItem(f"{minutes} min", minutes)
        self.fast_analysis_combo.setCurrentIndex(
            self.fast_analysis_combo.findData(10)
        )
        fast_controls_layout.addWidget(self.fast_analysis_combo)
        auto_layout.addLayout(fast_controls_layout)

        fast_info_layout = QGridLayout()
        fast_info_layout.addWidget(QLabel("Fast scope:"), 0, 0)
        self.fast_scope_label = QLabel("First 10 min / GPU-only", self)
        fast_info_layout.addWidget(self.fast_scope_label, 0, 1)
        fast_info_layout.addWidget(QLabel("Fast status:"), 1, 0)
        self.fast_status_label = QLabel("Ready", self)
        fast_info_layout.addWidget(self.fast_status_label, 1, 1)
        fast_info_layout.addWidget(QLabel("Fast elapsed:"), 2, 0)
        self.fast_elapsed_label = QLabel("--", self)
        fast_info_layout.addWidget(self.fast_elapsed_label, 2, 1)
        auto_layout.addLayout(fast_info_layout)

        fast_diag_layout = QGridLayout()
        fast_diag_layout.addWidget(QLabel("Fast confidence:"), 0, 0)
        self.fast_confidence_label = QLabel("-", self)
        fast_diag_layout.addWidget(self.fast_confidence_label, 0, 1)
        fast_diag_layout.addWidget(QLabel("Fast diagnostic:"), 1, 0)
        self.fast_diagnostic_status_label = QLabel("-", self)
        fast_diag_layout.addWidget(self.fast_diagnostic_status_label, 1, 1)
        fast_diag_layout.addWidget(QLabel("Best candidate:"), 2, 0)
        self.fast_best_candidate_label = QLabel("-", self)
        fast_diag_layout.addWidget(self.fast_best_candidate_label, 2, 1)
        fast_diag_layout.addWidget(QLabel("2nd candidate:"), 3, 0)
        self.fast_second_candidate_label = QLabel("-", self)
        fast_diag_layout.addWidget(self.fast_second_candidate_label, 3, 1)
        fast_diag_layout.addWidget(QLabel("Score margin:"), 4, 0)
        self.fast_margin_label = QLabel("-", self)
        fast_diag_layout.addWidget(self.fast_margin_label, 4, 1)
        fast_diag_layout.addWidget(QLabel("Sample agreement:"), 5, 0)
        self.fast_anchor_agreement_label = QLabel("-", self)
        fast_diag_layout.addWidget(self.fast_anchor_agreement_label, 5, 1)
        fast_diag_layout.addWidget(QLabel("Offset spread:"), 6, 0)
        self.fast_spread_label = QLabel("-", self)
        fast_diag_layout.addWidget(self.fast_spread_label, 6, 1)
        fast_diag_layout.addWidget(QLabel("Verification:"), 7, 0)
        self.fast_verification_label = QLabel("-", self)
        fast_diag_layout.addWidget(self.fast_verification_label, 7, 1)
        fast_diag_layout.addWidget(QLabel("Sample offsets:"), 8, 0)
        self.fast_sample_offsets_label = QLabel("-", self)
        self.fast_sample_offsets_label.setWordWrap(True)
        fast_diag_layout.addWidget(self.fast_sample_offsets_label, 8, 1)
        fast_diag_layout.addWidget(QLabel("Top 5:"), 9, 0)
        self.fast_top5_label = QLabel("-", self)
        self.fast_top5_label.setWordWrap(True)
        fast_diag_layout.addWidget(self.fast_top5_label, 9, 1)
        auto_layout.addLayout(fast_diag_layout)

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
        self.fast_sync_btn.clicked.connect(self._run_fast_auto_sync)
        self.fast_analysis_combo.currentIndexChanged.connect(
            self._on_fast_analysis_changed
        )
        self.app_state.fast_sync_analysis_changed.connect(
            self._on_fast_analysis_state_changed
        )
        self.sync_controller.sync_progress.connect(self._on_fast_sync_progress)
        self.sync_controller.sync_finished.connect(self._on_sync_finished)
        self.sync_controller.sync_error.connect(self._on_fast_sync_error)
        self.minus_1_btn.clicked.connect(lambda: self._adjust_offset(-1))
        self.plus_1_btn.clicked.connect(lambda: self._adjust_offset(1))
        self.minus_10_btn.clicked.connect(lambda: self._adjust_offset(-10))
        self.plus_10_btn.clicked.connect(lambda: self._adjust_offset(10))
        self.apply_offset_btn.clicked.connect(self._apply_manual_offset)
        self.lock_btn.clicked.connect(self._lock_offset)
        self.offset_comparison.offset_selected.connect(self._adjust_offset)

        # State observation
        self.app_state.sync_changed.connect(self._refresh_display)

    def _update_fast_scope_label(self, minutes: int):
        """Display the selected bounded analysis scope."""
        self.fast_scope_label.setText(f"First {int(minutes)} min / GPU-only")

    @Slot(int)
    def _on_fast_analysis_changed(self, index: int):
        """Persist the user-selected Fast Sync analysis duration."""
        minutes = self.fast_analysis_combo.itemData(index)
        if minutes is None:
            return
        minutes = int(minutes)
        self._update_fast_scope_label(minutes)
        self.app_state.set_fast_sync_analysis_minutes(minutes)

    @Slot(int)
    def _on_fast_analysis_state_changed(self, minutes: int):
        """Reflect a project-loaded Fast Sync duration in the combo box."""
        index = self.fast_analysis_combo.findData(int(minutes))
        if index >= 0 and index != self.fast_analysis_combo.currentIndex():
            self.fast_analysis_combo.setCurrentIndex(index)
        self._update_fast_scope_label(int(minutes))

    @Slot()
    def _run_auto_sync(self):
        """Invoke auto-sync via the controller."""
        self.auto_sync_btn.setEnabled(False)
        self.app_state.status_message.emit("Running auto-sync...")
        self.sync_controller.run_auto_sync()
        self.auto_sync_btn.setEnabled(True)

    @Slot()
    def _run_fast_auto_sync(self):
        """Invoke bounded Fast Auto Sync using the selected minute range."""
        minutes = int(self.fast_analysis_combo.currentData())
        self._update_fast_scope_label(minutes)
        self.fast_sync_btn.setEnabled(False)
        self.fast_status_label.setText(
            f"Starting GPU-only path ({minutes} min)..."
        )
        self.fast_elapsed_label.setText("0.0 s")
        if not self.sync_controller.run_fast_auto_sync(minutes):
            self.fast_sync_btn.setEnabled(True)
            self.fast_elapsed_label.setText("--")

    @Slot(object)
    def _on_fast_sync_progress(self, payload):
        """Display progress and elapsed time from Fast Auto Sync."""
        if not isinstance(payload, dict) or payload.get("strategy") != "fast":
            return
        message = payload.get("message")
        if message:
            self.fast_status_label.setText(str(message))
        elapsed = payload.get("elapsed_seconds")
        if elapsed is not None:
            self.fast_elapsed_label.setText(f"{float(elapsed):.1f} s")

    @Slot()
    def _on_sync_finished(self):
        """Re-enable the fast button after either sync worker finishes."""
        was_running = not self.fast_sync_btn.isEnabled()
        self.fast_sync_btn.setEnabled(True)
        if was_running and not self.fast_status_label.text().startswith("Error:"):
            self.fast_status_label.setText("Finished; review proposal")

    @Slot(str)
    def _on_fast_sync_error(self, message: str):
        """Keep Fast Auto Sync errors visible in the sync panel."""
        if self.fast_sync_btn.isEnabled() and not self.fast_status_label.text().startswith(
            "Starting"
        ):
            return
        self.fast_sync_btn.setEnabled(True)
        self.fast_status_label.setText(f"Error: {message}")
        self.fast_status_label.setToolTip(message)

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

    def _refresh_fast_diagnostics(self, proposal: SyncProposal):
        """Show bounded Fast Auto Sync evidence without changing lock state."""
        diagnostics = proposal.fast_diagnostics
        if not diagnostics:
            for label in (
                self.fast_confidence_label,
                self.fast_diagnostic_status_label,
                self.fast_best_candidate_label,
                self.fast_second_candidate_label,
                self.fast_margin_label,
                self.fast_anchor_agreement_label,
                self.fast_spread_label,
                self.fast_verification_label,
                self.fast_sample_offsets_label,
                self.fast_top5_label,
            ):
                label.setText("-")
            return

        fast_confidence = proposal.fast_confidence
        self.fast_confidence_label.setText(
            f"{float(fast_confidence):.3f}" if fast_confidence is not None else "-"
        )
        diagnostic_status = proposal.fast_diagnostic_status or "FAST_REVIEW"
        self.fast_diagnostic_status_label.setText(
            f"{diagnostic_status} (diagnostic only)"
        )
        self.fast_diagnostic_status_label.setStyleSheet(
            "color: #44cc44;"
            if diagnostic_status == "FAST_LOCKED"
            else "color: #ffaa00;"
        )

        best_offset = diagnostics.get("best_offset", "-")
        best_score = diagnostics.get("best_score")
        self.fast_best_candidate_label.setText(
            f"{best_offset} / {float(best_score):.3f}"
            if best_score is not None
            else str(best_offset)
        )
        second_offset = diagnostics.get("second_best_offset", "-")
        second_score = diagnostics.get("second_score")
        self.fast_second_candidate_label.setText(
            f"{second_offset} / {float(second_score):.3f}"
            if second_score is not None
            else str(second_offset)
        )
        margin = diagnostics.get("score_margin")
        self.fast_margin_label.setText(
            f"{float(margin):.3f}" if margin is not None else "-"
        )

        agreement_count = diagnostics.get("agreement_count", 0)
        independent_count = diagnostics.get("independent_sample_count", 0)
        agreement_percentage = diagnostics.get("agreement_percentage")
        self.fast_anchor_agreement_label.setText(
            f"{agreement_count}/{independent_count} "
            f"({float(agreement_percentage) * 100:.1f}%)"
            if agreement_percentage is not None
            else "-"
        )
        spread = diagnostics.get("offset_spread_frames")
        self.fast_spread_label.setText(
            f"{float(spread):.1f} frames" if spread is not None else "-"
        )

        verification = diagnostics.get("verification", {})
        post_score = verification.get("post_selection_score")
        valid_samples = verification.get("post_selection_valid_frame_samples", 0)
        independent_anchors = verification.get(
            "post_selection_independent_anchors", 0
        )
        self.fast_verification_label.setText(
            f"score={float(post_score):.3f}, "
            f"anchors={independent_anchors}, frames={valid_samples}"
            if post_score is not None
            else "-"
        )

        sample_text = []
        for sample in diagnostics.get("anchors", []):
            sample_text.append(
                f"S{sample.get('sample', '?')}="
                f"{sample.get('best_offset', '?')}"
            )
        self.fast_sample_offsets_label.setText(
            ", ".join(sample_text) if sample_text else "-"
        )

        candidate_text = []
        for candidate in diagnostics.get("candidates", [])[:5]:
            candidate_text.append(
                f"#{candidate.get('rank', '?')} "
                f"{candidate.get('offset', '?')} "
                f"({float(candidate.get('score', 0.0)):.3f})"
            )
        self.fast_top5_label.setText(" | ".join(candidate_text) or "-")

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
        self._refresh_fast_diagnostics(proposal)

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
