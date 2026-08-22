"""
Shot Lock Panel - Display and manage locked shots.

Shows all shots with their lock status. Fitting can only begin after
a shot's sync has been confirmed via LOCK. This panel displays:
- Shot list with lock status indicators
- Offset and confidence for each shot
- Fitting gate (shows which shots are ready for fitting)
- Ability to unlock/review shots
"""
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor

from gui.models.state import AppState
from gui.models.sync_state import ShotLock, LockStatus


class ShotLockPanel(QWidget):
    """Panel displaying shot lock status and fitting gate."""

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Create the panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Summary section
        summary_group = QGroupBox("Lock Summary", self)
        summary_layout = QHBoxLayout(summary_group)

        self.total_shots_label = QLabel("Total Shots: 0", self)
        self.locked_shots_label = QLabel("Locked: 0", self)
        self.unlocked_shots_label = QLabel("Unlocked: 0", self)
        self.review_shots_label = QLabel("Needs Review: 0", self)

        summary_layout.addWidget(self.total_shots_label)
        summary_layout.addWidget(self.locked_shots_label)
        summary_layout.addWidget(self.unlocked_shots_label)
        summary_layout.addWidget(self.review_shots_label)
        summary_layout.addStretch()

        main_layout.addWidget(summary_group)

        # Fitting gate
        gate_group = QGroupBox("Fitting Gate", self)
        gate_layout = QVBoxLayout(gate_group)
        self.gate_label = QLabel(
            "Fitting requires ALL shots to be LOCKED.\n"
            "Lock each shot's sync offset before proceeding.",
            self,
        )
        self.gate_label.setWordWrap(True)
        gate_layout.addWidget(self.gate_label)

        self.gate_status_label = QLabel("Gate: CLOSED", self)
        self.gate_status_label.setProperty("class", "gate-status")
        gate_layout.addWidget(self.gate_status_label)

        main_layout.addWidget(gate_group)

        # Shot table
        table_group = QGroupBox("Shot Locks", self)
        table_layout = QVBoxLayout(table_group)

        self.shot_table = QTableWidget(self)
        self.shot_table.setColumnCount(6)
        self.shot_table.setHorizontalHeaderLabels([
            "Shot ID",
            "Start Frame",
            "End Frame",
            "Offset",
            "Confidence",
            "Status",
        ])
        self.shot_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.shot_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.shot_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        table_layout.addWidget(self.shot_table)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.unlock_btn = QPushButton("Unlock Selected", self)
        self.review_btn = QPushButton("Mark for Review", self)
        self.lock_all_btn = QPushButton("Lock All (use current offset)", self)
        btn_layout.addWidget(self.unlock_btn)
        btn_layout.addWidget(self.review_btn)
        btn_layout.addWidget(self.lock_all_btn)
        btn_layout.addStretch()
        table_layout.addLayout(btn_layout)

        main_layout.addWidget(table_group, 1)

    def _connect_signals(self):
        """Connect signals."""
        self.unlock_btn.clicked.connect(self._unlock_selected)
        self.review_btn.clicked.connect(self._mark_review)
        self.lock_all_btn.clicked.connect(self._lock_all)

        # State observation
        self.app_state.shots_changed.connect(self._refresh_display)

    @Slot()
    def _unlock_selected(self):
        """Unlock the selected shot."""
        row = self.shot_table.currentRow()
        if 0 <= row < len(self.app_state.shot_locks):
            self.app_state.shot_locks[row].unlock()
            self.app_state.is_dirty = True
            self.app_state.shots_changed.emit()

    @Slot()
    def _mark_review(self):
        """Mark selected shot for review."""
        row = self.shot_table.currentRow()
        if 0 <= row < len(self.app_state.shot_locks):
            self.app_state.shot_locks[row].mark_needs_review()
            self.app_state.is_dirty = True
            self.app_state.shots_changed.emit()

    @Slot()
    def _lock_all(self):
        """Lock all shots using the current sync proposal offset."""
        proposal = self.app_state.sync_proposal
        if not proposal:
            return
        for shot_lock in self.app_state.shot_locks:
            if not shot_lock.is_locked:
                shot_lock.lock(proposal.offset, proposal.confidence)
        self.app_state.is_dirty = True
        self.app_state.shots_changed.emit()

    @Slot()
    def _refresh_display(self):
        """Refresh the shot table and summary from state."""
        locks = self.app_state.shot_locks

        # Update summary
        total = len(locks)
        locked = sum(1 for sl in locks if sl.lock_status == LockStatus.LOCKED)
        unlocked = sum(
            1 for sl in locks if sl.lock_status == LockStatus.UNLOCKED
        )
        review = sum(
            1 for sl in locks if sl.lock_status == LockStatus.NEEDS_REVIEW
        )

        self.total_shots_label.setText(f"Total Shots: {total}")
        self.locked_shots_label.setText(f"Locked: {locked}")
        self.unlocked_shots_label.setText(f"Unlocked: {unlocked}")
        self.review_shots_label.setText(f"Needs Review: {review}")

        # Update gate status
        if total > 0 and locked == total:
            self.gate_status_label.setText("Gate: OPEN - Ready for fitting")
            self.gate_status_label.setStyleSheet("color: #44cc44;")
        else:
            self.gate_status_label.setText(
                f"Gate: CLOSED - {total - locked} shot(s) need locking"
            )
            self.gate_status_label.setStyleSheet("color: #ff4444;")

        # Update table
        self.shot_table.setRowCount(total)
        for row, sl in enumerate(locks):
            self.shot_table.setItem(
                row, 0, QTableWidgetItem(sl.shot_id)
            )
            self.shot_table.setItem(
                row, 1, QTableWidgetItem(str(sl.hdr_start_frame))
            )
            self.shot_table.setItem(
                row, 2, QTableWidgetItem(str(sl.hdr_end_frame))
            )
            self.shot_table.setItem(
                row, 3, QTableWidgetItem(str(sl.offset))
            )
            self.shot_table.setItem(
                row, 4, QTableWidgetItem(f"{sl.confidence:.3f}")
            )

            # Status with color
            status_item = QTableWidgetItem(sl.lock_status.value.upper())
            if sl.lock_status == LockStatus.LOCKED:
                status_item.setForeground(QColor("#44cc44"))
            elif sl.lock_status == LockStatus.NEEDS_REVIEW:
                status_item.setForeground(QColor("#ffaa00"))
            else:
                status_item.setForeground(QColor("#aaaaaa"))
            self.shot_table.setItem(row, 5, status_item)
