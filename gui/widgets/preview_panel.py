"""
Preview Panel - Dual-source display with real frame extraction.

Shows HDR and OpenMatte sources with real video frames extracted via ffmpeg:
- 4 display modes: side-by-side, overlay, wiper, checkerboard
- Frame-accurate navigation: +/-1, +/-10, +/-1 second, timecode entry
- Zoom controls: Fit, 100% with scroll
- 3-frame cache (current +/- 1) to limit RAM
"""
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QSlider,
    QFrame,
    QScrollArea,
    QLineEdit,
)
from PySide6.QtCore import Qt, Slot, Signal, QSize, QTimer
from PySide6.QtGui import QPixmap, QPainter, QColor, QImage

from gui.controllers.pipeline_adapter import PREVIEW_DRAFT, PREVIEW_FULL
from gui.models.state import AppState


class PreviewMode:
    """Constants for preview display modes."""

    SIDE_BY_SIDE = "side_by_side"
    OVERLAY = "overlay"
    WIPER = "wiper"
    CHECKERBOARD = "checkerboard"


class FrameCache:
    """Simple LRU cache for extracted frames, limited to max_size entries.

    Stores QImage objects keyed by (source_path, frame_number).
    Limits to 3 entries per source (current frame +/- 1) = 6 total max.
    """

    def __init__(self, max_size: int = 6):
        self._max_size = max_size
        self._cache: dict = {}  # (path, frame) -> QImage
        self._order: list = []  # insertion order for LRU eviction

    def get(self, path: str, frame: int) -> Optional[QImage]:
        """Retrieve a cached frame or None."""
        key = (path, frame)
        image = self._cache.get(key)
        if image is not None and key in self._order:
            self._order.remove(key)
            self._order.append(key)
        return image

    def put(self, path: str, frame: int, image: QImage):
        """Store a frame image, evicting oldest if over max_size."""
        key = (path, frame)
        if key in self._cache:
            # Move to end (most recent)
            self._order.remove(key)
            self._order.append(key)
            self._cache[key] = image
            return

        # Evict if at capacity
        while len(self._cache) >= self._max_size:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = image
        self._order.append(key)

    def clear(self):
        """Clear all cached frames."""
        self._cache.clear()
        self._order.clear()


class CompositeDisplayWidget(QWidget):
    """Widget that renders frames in the selected display mode.

    Supports: side-by-side, overlay, wiper, and checkerboard compositing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)

        self._hdr_pixmap: Optional[QPixmap] = None
        self._om_pixmap: Optional[QPixmap] = None
        self._mode: str = PreviewMode.SIDE_BY_SIDE
        self._overlay_opacity: float = 0.5
        self._wiper_position: float = 0.5  # 0.0 to 1.0
        self._zoom_fit: bool = True
        self._zoom_level: float = 1.0
        self._hdr_label: str = "HDR"
        self._om_label: str = "OpenMatte"

    def set_hdr_frame(self, pixmap: Optional[QPixmap]):
        """Set the HDR frame pixmap."""
        self._hdr_pixmap = pixmap
        self.update()

    def set_om_frame(self, pixmap: Optional[QPixmap]):
        """Set the OpenMatte frame pixmap."""
        self._om_pixmap = pixmap
        self.update()

    def set_mode(self, mode: str):
        """Set the display composition mode."""
        self._mode = mode
        self.update()

    def set_overlay_opacity(self, opacity: float):
        """Set overlay blend opacity (0.0 - 1.0)."""
        self._overlay_opacity = max(0.0, min(1.0, opacity))
        self.update()

    def set_wiper_position(self, position: float):
        """Set wiper divider position (0.0 - 1.0)."""
        self._wiper_position = max(0.0, min(1.0, position))
        self.update()

    def set_zoom_fit(self, fit: bool):
        """Enable/disable fit-to-widget zoom mode."""
        self._zoom_fit = fit
        self.update()

    def set_zoom_level(self, level: float):
        """Set explicit zoom level (1.0 = 100%)."""
        self._zoom_level = level
        self._zoom_fit = False
        self.update()

    def set_labels(self, hdr_label: str, om_label: str):
        """Set labels for the two sources."""
        self._hdr_label = hdr_label
        self._om_label = om_label

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def paintEvent(self, event):
        """Render frames using the current display mode."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        widget_rect = self.rect()
        bg_color = QColor(30, 30, 30)
        painter.fillRect(widget_rect, bg_color)

        if self._hdr_pixmap is None and self._om_pixmap is None:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(widget_rect, Qt.AlignmentFlag.AlignCenter,
                             "No frames loaded\nLoad HDR and OM sources to preview")
            painter.end()
            return

        if self._mode == PreviewMode.SIDE_BY_SIDE:
            self._paint_side_by_side(painter, widget_rect)
        elif self._mode == PreviewMode.OVERLAY:
            self._paint_overlay(painter, widget_rect)
        elif self._mode == PreviewMode.WIPER:
            self._paint_wiper(painter, widget_rect)
        elif self._mode == PreviewMode.CHECKERBOARD:
            self._paint_checkerboard(painter, widget_rect)

        painter.end()

    def _scale_pixmap(self, pixmap: QPixmap, target_w: int, target_h: int) -> QPixmap:
        """Scale pixmap to fit within target dimensions, preserving aspect ratio."""
        if self._zoom_fit:
            return pixmap.scaled(
                target_w, target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            # Zoom level scaling
            new_w = int(pixmap.width() * self._zoom_level)
            new_h = int(pixmap.height() * self._zoom_level)
            return pixmap.scaled(
                new_w, new_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

    def _paint_side_by_side(self, painter: QPainter, rect):
        """Render HDR on left, OM on right, side by side."""
        half_w = rect.width() // 2
        full_h = rect.height()

        # Draw separator line
        painter.setPen(QColor(80, 80, 80))
        painter.drawLine(half_w, 0, half_w, full_h)

        # HDR (left)
        if self._hdr_pixmap:
            scaled = self._scale_pixmap(self._hdr_pixmap, half_w - 4, full_h - 20)
            x = (half_w - scaled.width()) // 2
            y = (full_h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor(100, 100, 100))
            from PySide6.QtCore import QRect
            painter.drawText(QRect(0, 0, half_w, full_h),
                             Qt.AlignmentFlag.AlignCenter, f"[{self._hdr_label}]\nNo frame")

        # OM (right)
        if self._om_pixmap:
            scaled = self._scale_pixmap(self._om_pixmap, half_w - 4, full_h - 20)
            x = half_w + (half_w - scaled.width()) // 2
            y = (full_h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor(100, 100, 100))
            from PySide6.QtCore import QRect
            painter.drawText(QRect(half_w, 0, half_w, full_h),
                             Qt.AlignmentFlag.AlignCenter, f"[{self._om_label}]\nNo frame")

        # Labels
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(8, 16, self._hdr_label)
        painter.drawText(half_w + 8, 16, self._om_label)

    def _paint_overlay(self, painter: QPainter, rect):
        """Render HDR with OM blended on top using adjustable opacity."""
        full_w = rect.width()
        full_h = rect.height()

        # Draw HDR at full opacity
        if self._hdr_pixmap:
            scaled = self._scale_pixmap(self._hdr_pixmap, full_w, full_h)
            x = (full_w - scaled.width()) // 2
            y = (full_h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        # Draw OM with opacity
        if self._om_pixmap:
            painter.setOpacity(self._overlay_opacity)
            scaled = self._scale_pixmap(self._om_pixmap, full_w, full_h)
            x = (full_w - scaled.width()) // 2
            y = (full_h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.setOpacity(1.0)

        # Label
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(8, 16, f"Overlay ({int(self._overlay_opacity * 100)}%)")

    def _paint_wiper(self, painter: QPainter, rect):
        """Render with a vertical wiper: left = HDR, right = OM."""
        full_w = rect.width()
        full_h = rect.height()
        split_x = int(full_w * self._wiper_position)

        # HDR on left side (clip to left of wiper)
        if self._hdr_pixmap:
            scaled = self._scale_pixmap(self._hdr_pixmap, full_w, full_h)
            x_offset = (full_w - scaled.width()) // 2
            y_offset = (full_h - scaled.height()) // 2
            painter.setClipRect(0, 0, split_x, full_h)
            painter.drawPixmap(x_offset, y_offset, scaled)

        # OM on right side (clip to right of wiper)
        if self._om_pixmap:
            scaled = self._scale_pixmap(self._om_pixmap, full_w, full_h)
            x_offset = (full_w - scaled.width()) // 2
            y_offset = (full_h - scaled.height()) // 2
            painter.setClipRect(split_x, 0, full_w - split_x, full_h)
            painter.drawPixmap(x_offset, y_offset, scaled)

        # Draw wiper line
        painter.setClipping(False)
        painter.setPen(QColor(255, 200, 0))
        painter.drawLine(split_x, 0, split_x, full_h)

        # Labels
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(8, 16, self._hdr_label)
        painter.drawText(split_x + 8, 16, self._om_label)

    def _paint_checkerboard(self, painter: QPainter, rect):
        """Render alternating 64x64 blocks from HDR and OM."""
        full_w = rect.width()
        full_h = rect.height()
        block_size = 64

        # Scale both to same size
        hdr_scaled = None
        om_scaled = None
        if self._hdr_pixmap:
            hdr_scaled = self._scale_pixmap(self._hdr_pixmap, full_w, full_h)
        if self._om_pixmap:
            om_scaled = self._scale_pixmap(self._om_pixmap, full_w, full_h)

        # Use image compositing for checkerboard
        if hdr_scaled is None and om_scaled is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No frames")
            return

        # Determine composited area size
        target_w = full_w
        target_h = full_h
        if hdr_scaled:
            target_w = hdr_scaled.width()
            target_h = hdr_scaled.height()
        elif om_scaled:
            target_w = om_scaled.width()
            target_h = om_scaled.height()

        x_off = (full_w - target_w) // 2
        y_off = (full_h - target_h) // 2

        # Draw block by block
        for row in range(0, target_h, block_size):
            for col in range(0, target_w, block_size):
                # Determine which source for this block
                block_row = row // block_size
                block_col = col // block_size
                use_hdr = (block_row + block_col) % 2 == 0

                bw = min(block_size, target_w - col)
                bh = min(block_size, target_h - row)

                source = hdr_scaled if use_hdr else om_scaled
                if source is None:
                    source = om_scaled if use_hdr else hdr_scaled
                if source is None:
                    continue

                # Copy block region
                block_pixmap = source.copy(col, row, bw, bh)
                painter.drawPixmap(x_off + col, y_off + row, block_pixmap)

        # Label
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(8, 16, "Checkerboard (64x64)")


class PreviewPanel(QWidget):
    """Panel for dual-source preview with real frame extraction."""

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._current_mode = PreviewMode.SIDE_BY_SIDE
        self._zoom_fit_mode = True
        self._zoom_level = 1.0
        self._pipeline_adapter = None  # Set via set_pipeline_adapter()

        # Frame cache: max 6 entries (3 per source: current +/- 1)
        self._frame_cache = FrameCache(max_size=6)
        self._preview_generation = 0
        self._is_playing = False
        self._scrubbing = False
        self._displayed_frame = -1
        self._displayed_quality = None
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(33)
        # Escalation to full quality happens only after the user settles.
        self._escalate_timer = QTimer(self)
        self._escalate_timer.setSingleShot(True)
        self._escalate_timer.setInterval(220)

        self._setup_ui()
        self._connect_signals()

    def set_pipeline_adapter(self, adapter):
        """Set the adapter used by the background preview decoder."""
        self._pipeline_adapter = adapter
        adapter.preview_frame_ready.connect(self._on_preview_frame_ready)
        adapter.preview_error.connect(self._on_preview_error)

    def _setup_ui(self):
        """Create the panel layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar: display mode and zoom
        toolbar_layout = QHBoxLayout()

        # Display mode selector
        toolbar_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("Side by Side", PreviewMode.SIDE_BY_SIDE)
        self.mode_combo.addItem("Overlay", PreviewMode.OVERLAY)
        self.mode_combo.addItem("Wiper", PreviewMode.WIPER)
        self.mode_combo.addItem("Checkerboard", PreviewMode.CHECKERBOARD)
        toolbar_layout.addWidget(self.mode_combo)

        toolbar_layout.addSpacing(10)

        # Overlay opacity slider (visible only in overlay mode)
        self.opacity_label = QLabel("Opacity:", self)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setMaximumWidth(120)
        self.opacity_value_label = QLabel("50%", self)
        toolbar_layout.addWidget(self.opacity_label)
        toolbar_layout.addWidget(self.opacity_slider)
        toolbar_layout.addWidget(self.opacity_value_label)
        self.opacity_label.setVisible(False)
        self.opacity_slider.setVisible(False)
        self.opacity_value_label.setVisible(False)

        toolbar_layout.addSpacing(20)

        # Zoom controls
        toolbar_layout.addWidget(QLabel("Zoom:"))
        self.zoom_fit_btn = QPushButton("Fit", self)
        self.zoom_100_btn = QPushButton("100%", self)
        self.zoom_in_btn = QPushButton("+", self)
        self.zoom_out_btn = QPushButton("-", self)
        self.zoom_label = QLabel("Fit", self)
        toolbar_layout.addWidget(self.zoom_fit_btn)
        toolbar_layout.addWidget(self.zoom_100_btn)
        toolbar_layout.addWidget(self.zoom_out_btn)
        toolbar_layout.addWidget(self.zoom_in_btn)
        toolbar_layout.addWidget(self.zoom_label)

        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        # Scroll area for zoom 100% mode
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        # Composite display widget (renders all modes via QPainter)
        self.display_widget = CompositeDisplayWidget(self)
        self.scroll_area.setWidget(self.display_widget)

        main_layout.addWidget(self.scroll_area, 1)

        # Wiper slider (only visible in wiper mode)
        self.wiper_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.wiper_slider.setRange(0, 100)
        self.wiper_slider.setValue(50)
        self.wiper_slider.setVisible(False)
        main_layout.addWidget(self.wiper_slider)

        # Frame navigation
        nav_group = QGroupBox("Frame Navigation", self)
        nav_layout = QVBoxLayout(nav_group)

        # Navigation buttons row 1: playback and frame steps
        nav_row1 = QHBoxLayout()
        self.play_pause_btn = QPushButton("Play", self)
        self.play_pause_btn.setMinimumWidth(72)
        self.preview_status_label = QLabel("Idle", self)
        self.prev_1sec_btn = QPushButton("-1s", self)
        self.prev_10_btn = QPushButton("-10", self)
        self.prev_1_btn = QPushButton("-1", self)
        self.next_1_btn = QPushButton("+1", self)
        self.next_10_btn = QPushButton("+10", self)
        self.next_1sec_btn = QPushButton("+1s", self)

        nav_row1.addWidget(self.play_pause_btn)
        nav_row1.addWidget(self.preview_status_label)
        nav_row1.addWidget(self.prev_1sec_btn)
        nav_row1.addWidget(self.prev_10_btn)
        nav_row1.addWidget(self.prev_1_btn)
        nav_row1.addWidget(self.next_1_btn)
        nav_row1.addWidget(self.next_10_btn)
        nav_row1.addWidget(self.next_1sec_btn)
        nav_layout.addLayout(nav_row1)

        # Navigation row 2: frame number and timecode
        nav_row2 = QHBoxLayout()
        nav_row2.addWidget(QLabel("Frame:"))
        self.frame_spinbox = QSpinBox(self)
        self.frame_spinbox.setRange(0, 999999)
        self.frame_spinbox.setValue(0)
        nav_row2.addWidget(self.frame_spinbox)

        nav_row2.addSpacing(20)
        nav_row2.addWidget(QLabel("Timecode:"))
        self.timecode_edit = QLineEdit("00:00:00:00", self)
        self.timecode_edit.setMaximumWidth(120)
        nav_row2.addWidget(self.timecode_edit)
        self.go_to_tc_btn = QPushButton("Go", self)
        nav_row2.addWidget(self.go_to_tc_btn)

        nav_row2.addStretch()
        nav_layout.addLayout(nav_row2)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        nav_layout.addWidget(self.frame_slider)

        main_layout.addWidget(nav_group)

    def _connect_signals(self):
        """Connect UI signals."""
        # Mode selection
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        # Overlay opacity
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)

        # Wiper position
        self.wiper_slider.valueChanged.connect(self._on_wiper_changed)

        # Zoom
        self.zoom_fit_btn.clicked.connect(self._zoom_fit)
        self.zoom_100_btn.clicked.connect(self._zoom_100)
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        self.zoom_out_btn.clicked.connect(self._zoom_out)

        # Navigation
        self.play_pause_btn.clicked.connect(self._toggle_playback)
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._escalate_timer.timeout.connect(self._request_full_quality)
        self.frame_slider.sliderPressed.connect(self._on_scrub_started)
        self.frame_slider.sliderReleased.connect(self._on_scrub_finished)
        self.prev_1_btn.clicked.connect(lambda: self._navigate(-1))
        self.next_1_btn.clicked.connect(lambda: self._navigate(1))
        self.prev_10_btn.clicked.connect(lambda: self._navigate(-10))
        self.next_10_btn.clicked.connect(lambda: self._navigate(10))
        self.prev_1sec_btn.clicked.connect(lambda: self._navigate_seconds(-1))
        self.next_1sec_btn.clicked.connect(lambda: self._navigate_seconds(1))
        self.frame_spinbox.valueChanged.connect(self._on_frame_spinbox_changed)
        self.go_to_tc_btn.clicked.connect(self._go_to_timecode)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)

        # State observation
        self.app_state.preview_frame_changed.connect(self._refresh_display)
        self.app_state.sources_changed.connect(self._on_sources_changed)
        self.app_state.sync_changed.connect(self._on_sync_changed)

    @Slot(int)
    def _on_mode_changed(self, index: int):
        """Handle display mode change."""
        self._current_mode = self.mode_combo.itemData(index)
        self.display_widget.set_mode(self._current_mode)

        # Show/hide wiper slider
        self.wiper_slider.setVisible(self._current_mode == PreviewMode.WIPER)

        # Show/hide opacity controls
        is_overlay = self._current_mode == PreviewMode.OVERLAY
        self.opacity_label.setVisible(is_overlay)
        self.opacity_slider.setVisible(is_overlay)
        self.opacity_value_label.setVisible(is_overlay)

    @Slot(int)
    def _on_opacity_changed(self, value: int):
        """Handle overlay opacity slider change."""
        opacity = value / 100.0
        self.display_widget.set_overlay_opacity(opacity)
        self.opacity_value_label.setText(f"{value}%")

    @Slot(int)
    def _on_wiper_changed(self, value: int):
        """Handle wiper position slider change."""
        position = value / 100.0
        self.display_widget.set_wiper_position(position)

    @Slot()
    def _zoom_fit(self):
        """Set zoom to fit view."""
        self._zoom_fit_mode = True
        self._zoom_level = 1.0
        self.zoom_label.setText("Fit")
        self.display_widget.set_zoom_fit(True)
        self.scroll_area.setWidgetResizable(True)

    @Slot()
    def _zoom_100(self):
        """Set zoom to 100% (1:1 pixel mapping)."""
        self._zoom_fit_mode = False
        self._zoom_level = 1.0
        self.zoom_label.setText("100%")
        self.display_widget.set_zoom_level(1.0)
        self.scroll_area.setWidgetResizable(False)
        # Resize widget to accommodate 1:1
        self.display_widget.setMinimumSize(960, 540)

    @Slot()
    def _zoom_in(self):
        """Zoom in."""
        self._zoom_level = min(self._zoom_level * 1.25, 8.0)
        self._zoom_fit_mode = False
        self.zoom_label.setText(f"{int(self._zoom_level * 100)}%")
        self.display_widget.set_zoom_level(self._zoom_level)
        self.scroll_area.setWidgetResizable(False)
        new_w = int(960 * self._zoom_level)
        new_h = int(540 * self._zoom_level)
        self.display_widget.setMinimumSize(new_w, new_h)

    @Slot()
    def _zoom_out(self):
        """Zoom out."""
        self._zoom_level = max(self._zoom_level / 1.25, 0.25)
        self._zoom_fit_mode = False
        self.zoom_label.setText(f"{int(self._zoom_level * 100)}%")
        self.display_widget.set_zoom_level(self._zoom_level)
        self.scroll_area.setWidgetResizable(False)
        new_w = int(960 * self._zoom_level)
        new_h = int(540 * self._zoom_level)
        self.display_widget.setMinimumSize(new_w, new_h)

    def _navigate(self, delta: int):
        """Navigate by frame delta."""
        new_frame = max(0, self.app_state.current_frame + delta)
        self.app_state.set_current_frame(new_frame)

    def _navigate_seconds(self, seconds: int):
        """Navigate by seconds (uses source FPS)."""
        fps = 24.0  # Default
        if self.app_state.hdr_source and self.app_state.hdr_source.fps > 0:
            fps = self.app_state.hdr_source.fps
        delta = int(seconds * fps)
        self._navigate(delta)

    @Slot(int)
    def _on_frame_spinbox_changed(self, value: int):
        """Handle direct frame number entry."""
        self.app_state.set_current_frame(value)

    @Slot()
    def _go_to_timecode(self):
        """Navigate to timecode entry."""
        tc = self.timecode_edit.text().strip()
        frame = self._timecode_to_frame(tc)
        if frame >= 0:
            self.app_state.set_current_frame(frame)

    @Slot(int)
    def _on_slider_changed(self, value: int):
        """Handle a real frame-number slider without blocking the GUI."""
        total = self._get_total_frames()
        if total > 0:
            self.app_state.set_current_frame(
                max(0, min(total - 1, int(value)))
            )

    @Slot()
    def _on_sources_changed(self):
        """Reset preview navigation when the source pair changes."""
        self._stop_playback()
        self._escalate_timer.stop()
        self._preview_generation += 1
        self._frame_cache.clear()
        self._displayed_frame = -1
        self._displayed_quality = None
        total = self._get_total_frames()
        maximum = max(0, total - 1)
        self.frame_spinbox.setRange(0, maximum)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, maximum)
        self.frame_slider.setValue(min(self.app_state.current_frame, maximum))
        self.frame_slider.blockSignals(False)
        self._refresh_display()

    @Slot()
    def _on_sync_changed(self):
        """Request the same HDR frame with the newly proposed OM offset."""
        self._preview_generation += 1
        self._refresh_display()

    @Slot()
    def _toggle_playback(self):
        """Start or stop timer-driven playback."""
        if self._is_playing:
            self._stop_playback()
            return
        if self._get_total_frames() <= 0 or self._pipeline_adapter is None:
            self.preview_status_label.setText("Load HDR source")
            return
        fps = self._get_fps()
        self._playback_timer.setInterval(max(10, int(round(1000.0 / fps))))
        self._is_playing = True
        self.play_pause_btn.setText("Pause")
        self.preview_status_label.setText("Playing")
        self._playback_timer.start()

    def _stop_playback(self):
        """Stop playback and escalate the paused frame to full quality."""
        was_playing = self._is_playing
        self._is_playing = False
        self._playback_timer.stop()
        if hasattr(self, "play_pause_btn"):
            self.play_pause_btn.setText("Play")
        if was_playing and not self._scrubbing and self._pipeline_adapter is not None:
            self._escalate_timer.start()

    @Slot()
    def _on_playback_tick(self):
        """Advance one frame; decoding remains in PreviewWorker."""
        total = self._get_total_frames()
        next_frame = self.app_state.current_frame + 1
        if total <= 0 or next_frame >= total:
            self._stop_playback()
            return
        self.app_state.set_current_frame(next_frame)

    @Slot(object)
    def _on_preview_frame_ready(self, result):
        """Apply only the newest completed background decode."""
        if result.get("generation") != self._preview_generation:
            return
        frame = int(result.get("frame", -1))
        if frame != self.app_state.current_frame:
            return

        quality = result.get("quality", PREVIEW_DRAFT)
        # Never replace an already displayed full-quality frame with a draft.
        if (
            quality == PREVIEW_DRAFT
            and frame == self._displayed_frame
            and self._displayed_quality == PREVIEW_FULL
        ):
            return

        hdr_image = result.get("hdr_image")
        om_image = result.get("om_image")
        om_frame = int(result.get("om_frame", frame))
        # Only full-quality frames are cached, so the cache can never satisfy a
        # request with a lower-resolution image.
        if quality == PREVIEW_FULL:
            if self.app_state.hdr_source and hdr_image is not None:
                self._frame_cache.put(
                    self.app_state.hdr_source.path, frame, hdr_image
                )
            if self.app_state.om_source and om_image is not None:
                self._frame_cache.put(
                    self.app_state.om_source.path, om_frame, om_image
                )

        self.display_widget.set_hdr_frame(
            QPixmap.fromImage(hdr_image) if hdr_image is not None else None
        )
        self.display_widget.set_om_frame(
            QPixmap.fromImage(om_image) if om_image is not None else None
        )
        self._displayed_frame = frame
        self._displayed_quality = quality
        self._update_quality_badge(quality)

    @Slot(str)
    def _on_preview_error(self, message: str):
        """Show worker errors without stopping the rest of the GUI."""
        self.preview_status_label.setText("Preview unavailable")
        self.app_state.status_message.emit(message)

    @Slot()
    def _refresh_display(self):
        """Show a fast draft immediately, then escalate to full quality."""
        frame = self.app_state.current_frame
        total = self._get_total_frames()
        if total > 0:
            frame = max(0, min(total - 1, frame))
            if frame != self.app_state.current_frame:
                self.app_state.current_frame = frame

        self._sync_navigation_controls(frame)
        self._set_display_labels()

        om_frame = self._om_frame_for(frame)

        # A cached pair is always full quality, so it can be shown directly.
        if self._show_cached_pair(frame, om_frame):
            return

        if self._pipeline_adapter is None or not self._has_any_source():
            self.preview_status_label.setText("Load HDR and OM sources")
            return

        self._escalate_timer.stop()
        self._pipeline_adapter.cancel_full_preview()
        self._request_frames(frame, om_frame, PREVIEW_DRAFT, self._draft_width())
        self.preview_status_label.setText(f"Frame {frame}")

        # Playback keeps the draft lane busy; escalate only once it settles.
        if not self._is_playing and not self._scrubbing:
            self._escalate_timer.start()

    @Slot()
    def _request_full_quality(self):
        """Ask for the settled frame at full preview resolution."""
        if self._pipeline_adapter is None or self._is_playing or self._scrubbing:
            return
        if not self._has_any_source():
            return
        frame = self.app_state.current_frame
        self._request_frames(
            frame, self._om_frame_for(frame), PREVIEW_FULL, self._full_width()
        )

    def _request_frames(self, frame: int, om_frame: int, quality: str, width: int):
        self._pipeline_adapter.request_preview_frames(
            self._source_request(self.app_state.hdr_source, width),
            self._source_request(self.app_state.om_source, width),
            frame,
            om_frame,
            self._preview_generation,
            preview_width=width,
            quality=quality,
        )

    def _om_frame_for(self, frame: int) -> int:
        """Apply the shot-level sync offset to get the OM frame."""
        offset = 0
        if self.app_state.sync_proposal:
            offset = self.app_state.sync_proposal.offset
        return max(0, frame + offset)

    def _has_any_source(self) -> bool:
        return bool(
            (self.app_state.hdr_source and self.app_state.hdr_source.path)
            or (self.app_state.om_source and self.app_state.om_source.path)
        )

    def _show_cached_pair(self, frame: int, om_frame: int) -> bool:
        """Display a cached full-quality pair if both frames are present."""
        if not (self.app_state.hdr_source and self.app_state.om_source):
            return False
        cached_hdr = self._frame_cache.get(self.app_state.hdr_source.path, frame)
        cached_om = self._frame_cache.get(self.app_state.om_source.path, om_frame)
        if cached_hdr is None or cached_om is None:
            return False
        self.display_widget.set_hdr_frame(QPixmap.fromImage(cached_hdr))
        self.display_widget.set_om_frame(QPixmap.fromImage(cached_om))
        self._displayed_frame = frame
        self._displayed_quality = PREVIEW_FULL
        self._update_quality_badge(PREVIEW_FULL)
        return True

    def _draft_width(self) -> int:
        """Small target for interaction; smaller still while dragging."""
        return 480 if self._scrubbing else 720

    def _full_width(self) -> int:
        """Target the visible area, capped by the source resolution."""
        if self._zoom_fit_mode:
            width = max(640, self.display_widget.width())
            if self._current_mode == PreviewMode.SIDE_BY_SIDE:
                width = max(640, width // 2)
        else:
            width = int(1920 * max(self._zoom_level, 1.0))
        source = self.app_state.hdr_source or self.app_state.om_source
        if source and source.width > 0:
            width = min(width, source.width)
        return max(2, int(width) // 2 * 2)

    def _update_quality_badge(self, quality: str):
        if quality == PREVIEW_FULL:
            self.preview_status_label.setText("Full")
            self.preview_status_label.setStyleSheet("color: #44cc44;")
        else:
            self.preview_status_label.setText("Draft")
            self.preview_status_label.setStyleSheet("color: #ffaa00;")

    @Slot()
    def _on_scrub_started(self):
        """Drop to draft quality while the timeline is being dragged."""
        self._scrubbing = True
        self._escalate_timer.stop()
        if self._pipeline_adapter is not None:
            self._pipeline_adapter.cancel_full_preview()

    @Slot()
    def _on_scrub_finished(self):
        """Escalate to full quality once the timeline is released."""
        self._scrubbing = False
        if not self._is_playing:
            self._escalate_timer.start()

    def _source_request(self, source, preview_width: int):
        """Convert GUI source metadata to a worker request."""
        if source is None or not source.path:
            return None
        return {
            "path": source.path,
            "width": source.width,
            "height": source.height,
            "fps": source.fps,
            "preview_width": preview_width,
        }

    def _set_display_labels(self):
        hdr_label = "HDR"
        om_label = "OpenMatte"
        if self.app_state.hdr_source:
            hdr_label = f"HDR: {self.app_state.hdr_source.filename}"
        if self.app_state.om_source:
            om_label = f"OM: {self.app_state.om_source.filename}"
        self.display_widget.set_labels(hdr_label, om_label)

    def _sync_navigation_controls(self, frame: int):
        """Keep frame number, timecode, and slider synchronized."""
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame)
        self.frame_spinbox.blockSignals(False)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame)
        self.frame_slider.blockSignals(False)
        self.timecode_edit.setText(self._frame_to_timecode(frame))

    def _get_fps(self) -> float:
        if self.app_state.hdr_source and self.app_state.hdr_source.fps > 0:
            return self.app_state.hdr_source.fps
        return 24.0

    def _get_total_frames(self) -> int:
        """Get total frame count from HDR source."""
        if self.app_state.hdr_source:
            return self.app_state.hdr_source.frame_count
        return 0

    def _frame_to_timecode(self, frame: int) -> str:
        """Convert frame number to timecode string."""
        fps = 24.0
        if self.app_state.hdr_source and self.app_state.hdr_source.fps > 0:
            fps = self.app_state.hdr_source.fps

        total_seconds = frame / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        frames = int(frame % fps)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def _timecode_to_frame(self, tc: str) -> int:
        """Convert timecode string to frame number."""
        fps = 24.0
        if self.app_state.hdr_source and self.app_state.hdr_source.fps > 0:
            fps = self.app_state.hdr_source.fps

        try:
            parts = tc.split(":")
            if len(parts) == 4:
                h, m, s, f = [int(p) for p in parts]
                total_seconds = h * 3600 + m * 60 + s
                return int(total_seconds * fps) + f
        except (ValueError, IndexError):
            pass
        return -1
