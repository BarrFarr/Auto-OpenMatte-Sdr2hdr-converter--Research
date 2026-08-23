"""
Main window with tabbed panel layout for the Auto OpenMatte GUI.

The window organizes all panels into a logical workflow:
1. Source - select and validate input files
2. Sync - propose and lock frame alignment
3. Preview - dual-source frame inspection
4. Shot Lock - persist shot-level sync decisions
5. Output Preview - view processed output
6. Quality - diagnostic metrics
7. Render - output configuration and monitoring
"""
from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QStatusBar,
    QMenuBar,
    QMenu,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QFileDialog,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QKeySequence

from gui import __app_name__, __version__
from gui.models.state import AppState
from gui.widgets.source_panel import SourcePanel
from gui.widgets.sync_panel import SyncPanel
from gui.widgets.preview_panel import PreviewPanel
from gui.widgets.shot_lock_panel import ShotLockPanel
from gui.widgets.output_preview_panel import OutputPreviewPanel
from gui.widgets.quality_panel import QualityPanel
from gui.widgets.render_panel import RenderPanel
from gui.controllers.pipeline_adapter import PipelineAdapter
from gui.controllers.sync_controller import SyncController
from gui.controllers.render_controller import RenderController
from gui.controllers.preview_controller import PreviewController


class MainWindow(QMainWindow):
    """Main application window containing all workflow panels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{__app_name__} v{__version__}")
        self.setMinimumSize(1400, 900)

        # Application state
        self.app_state = AppState()

        # Controllers
        self.pipeline_adapter = PipelineAdapter(self.app_state)
        self.sync_controller = SyncController(self.app_state, self.pipeline_adapter)
        self.render_controller = RenderController(self.app_state, self.pipeline_adapter)
        self.preview_controller = PreviewController(self.app_state, self)

        # Setup UI
        self._setup_menu_bar()
        self._setup_central_widget()
        self._setup_status_bar()
        self._connect_signals()

    def _setup_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # File menu
        file_menu = QMenu("&File", self)
        menu_bar.addMenu(file_menu)

        self.action_new_project = QAction("&New Project", self)
        self.action_new_project.setShortcut(QKeySequence.StandardKey.New)
        file_menu.addAction(self.action_new_project)

        self.action_open_project = QAction("&Open Project...", self)
        self.action_open_project.setShortcut(QKeySequence.StandardKey.Open)
        file_menu.addAction(self.action_open_project)

        self.action_save_project = QAction("&Save Project", self)
        self.action_save_project.setShortcut(QKeySequence.StandardKey.Save)
        file_menu.addAction(self.action_save_project)

        self.action_save_as = QAction("Save &As...", self)
        self.action_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        file_menu.addAction(self.action_save_as)

        file_menu.addSeparator()

        self.action_exit = QAction("E&xit", self)
        self.action_exit.setShortcut(QKeySequence("Alt+F4"))
        file_menu.addAction(self.action_exit)

        # View menu
        view_menu = QMenu("&View", self)
        menu_bar.addMenu(view_menu)

        self.action_reset_layout = QAction("&Reset Layout", self)
        view_menu.addAction(self.action_reset_layout)

        # Help menu
        help_menu = QMenu("&Help", self)
        menu_bar.addMenu(help_menu)

        self.action_about = QAction("&About", self)
        help_menu.addAction(self.action_about)

    def _setup_central_widget(self):
        """Create the tabbed central widget with all panels."""
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tab_widget = QTabWidget(central)
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setDocumentMode(True)

        # Create panels
        self.source_panel = SourcePanel(
            self.app_state, self.pipeline_adapter, self
        )
        self.sync_panel = SyncPanel(
            self.app_state, self.sync_controller, self
        )
        self.preview_panel = PreviewPanel(self.app_state, self)
        self.preview_panel.set_pipeline_adapter(self.pipeline_adapter)
        self.preview_panel.set_preview_controller(self.preview_controller)
        self.shot_lock_panel = ShotLockPanel(self.app_state, self)
        self.output_preview_panel = OutputPreviewPanel(self.app_state, self)
        self.quality_panel = QualityPanel(self.app_state, self)
        self.render_panel = RenderPanel(
            self.app_state, self.render_controller, self
        )

        # Add tabs
        self.tab_widget.addTab(self.source_panel, "1. Source")
        self.tab_widget.addTab(self.sync_panel, "2. Sync")
        self.tab_widget.addTab(self.preview_panel, "3. Preview")
        self.tab_widget.addTab(self.shot_lock_panel, "4. Shot Lock")
        self.tab_widget.addTab(self.output_preview_panel, "5. Output")
        self.tab_widget.addTab(self.quality_panel, "6. Quality")
        self.tab_widget.addTab(self.render_panel, "7. Render")

        layout.addWidget(self.tab_widget)
        self.setCentralWidget(central)

    def _setup_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _connect_signals(self):
        """Connect menu actions and inter-panel signals."""
        self.action_new_project.triggered.connect(self._on_new_project)
        self.action_open_project.triggered.connect(self._on_open_project)
        self.action_save_project.triggered.connect(self._on_save_project)
        self.action_save_as.triggered.connect(self._on_save_project_as)
        self.action_exit.triggered.connect(self.close)
        self.action_about.triggered.connect(self._on_about)

        # State change propagation
        self.app_state.project_changed.connect(self._on_project_changed)
        self.app_state.status_message.connect(self.status_bar.showMessage)

    @Slot()
    def _on_new_project(self):
        """Create a new empty project."""
        self.app_state.new_project()
        self.status_bar.showMessage("New project created")

    @Slot()
    def _on_open_project(self):
        """Open an existing .omhdr project file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "OpenMatte HDR Projects (*.omhdr);;All Files (*)",
        )
        if path:
            success = self.app_state.load_project(path)
            if success:
                self.status_bar.showMessage(f"Loaded: {path}")
            else:
                QMessageBox.warning(
                    self, "Error", f"Failed to load project: {path}"
                )

    @Slot()
    def _on_save_project(self):
        """Save current project."""
        if self.app_state.project_path:
            success = self.app_state.save_project()
            if success:
                self.status_bar.showMessage(
                    f"Saved: {self.app_state.project_path}"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to save project")
        else:
            self._on_save_project_as()

    @Slot()
    def _on_save_project_as(self):
        """Save project to a new file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "OpenMatte HDR Projects (*.omhdr);;All Files (*)",
        )
        if path:
            if not path.endswith(".omhdr"):
                path += ".omhdr"
            success = self.app_state.save_project(path)
            if success:
                self.status_bar.showMessage(f"Saved: {path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save project")

    @Slot()
    def _on_about(self):
        """Show About dialog."""
        QMessageBox.about(
            self,
            f"About {__app_name__}",
            f"{__app_name__} v{__version__}\n\n"
            "GUI control surface for the Auto OpenMatte\n"
            "SDR-to-HDR video processing pipeline.\n\n"
            "This application orchestrates the existing backend.\n"
            "It does not implement new processing algorithms.",
        )

    @Slot()
    def _on_project_changed(self):
        """Update window title when project changes."""
        title = f"{__app_name__} v{__version__}"
        if self.app_state.project_path:
            title += f" - {self.app_state.project_path}"
        if self.app_state.is_dirty:
            title += " *"
        self.setWindowTitle(title)

    def closeEvent(self, event):
        """Handle window close with unsaved changes check."""
        should_close = True
        if self.app_state.is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "There are unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_project()
            elif reply == QMessageBox.StandardButton.Cancel:
                should_close = False
        if should_close:
            self.preview_controller.close()
            event.accept()
        else:
            event.ignore()
