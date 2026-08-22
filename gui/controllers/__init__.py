"""
GUI controllers - business logic adapters connecting GUI to backend.

Controllers do NOT implement processing logic. They adapt and orchestrate
calls to the existing auto_openmatte backend.
"""
from gui.controllers.pipeline_adapter import PipelineAdapter
from gui.controllers.sync_controller import SyncController
from gui.controllers.render_controller import RenderController

__all__ = ["PipelineAdapter", "SyncController", "RenderController"]
