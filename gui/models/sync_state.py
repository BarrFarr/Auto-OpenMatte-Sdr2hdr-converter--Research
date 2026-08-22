"""
Sync state models for the GUI.

These represent the GUI's view of sync results - the auto-sync output is
treated as a PROPOSAL that the user must confirm before it becomes a lock.

SyncProposal: Result from auto-sync (offset, score, confidence, quality status)
ShotLock: User-confirmed sync decision for a specific shot
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SyncQuality(Enum):
    """Quality assessment of the sync proposal.

    These are displayed to the user so they can make an informed
    decision about whether to accept the proposed offset.
    """

    STABLE = "stable"
    UNCERTAIN = "uncertain"
    AMBIGUOUS = "ambiguous"


class LockStatus(Enum):
    """Status of a shot lock."""

    UNLOCKED = "unlocked"
    LOCKED = "locked"
    NEEDS_REVIEW = "needs_review"


@dataclass
class SyncProposal:
    """Auto-sync result presented as a proposal, not a final decision.

    The GUI shows offset N-1, N, and N+1 simultaneously so the user
    can visually verify which alignment is correct.

    Attributes:
        offset: Proposed frame offset (OM_frame = HDR_frame + offset)
        score: Raw correlation score from the sync algorithm
        confidence: Normalized confidence 0.0-1.0
        quality: Assessment of sync reliability (stable/uncertain/ambiguous)
        offset_minus_1: offset - 1 for comparison display
        offset_plus_1: offset + 1 for comparison display
        is_accepted: Whether user has accepted this proposal
    """

    offset: int = 0
    score: float = 0.0
    confidence: float = 0.0
    quality: SyncQuality = SyncQuality.UNCERTAIN
    offset_minus_1: int = -1
    offset_plus_1: int = 1
    is_accepted: bool = False

    def __post_init__(self):
        """Ensure N-1 and N+1 are consistent with offset."""
        self.offset_minus_1 = self.offset - 1
        self.offset_plus_1 = self.offset + 1

    def adjust_offset(self, delta: int):
        """Adjust the proposed offset by a delta value.

        Args:
            delta: Frame offset adjustment (+/- value)
        """
        self.offset += delta
        self.offset_minus_1 = self.offset - 1
        self.offset_plus_1 = self.offset + 1

    def set_offset(self, new_offset: int):
        """Set an explicit offset value (manual entry).

        Args:
            new_offset: The new frame offset value
        """
        self.offset = new_offset
        self.offset_minus_1 = self.offset - 1
        self.offset_plus_1 = self.offset + 1

    def to_dict(self) -> dict:
        """Serialize to dictionary for project save."""
        return {
            "offset": self.offset,
            "score": self.score,
            "confidence": self.confidence,
            "quality": self.quality.value,
            "is_accepted": self.is_accepted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SyncProposal":
        """Deserialize from dictionary."""
        quality = SyncQuality(data.get("quality", "uncertain"))
        return cls(
            offset=data.get("offset", 0),
            score=data.get("score", 0.0),
            confidence=data.get("confidence", 0.0),
            quality=quality,
            is_accepted=data.get("is_accepted", False),
        )


@dataclass
class ShotLock:
    """User-confirmed sync decision for a specific shot.

    Once locked, the fitting process can begin for this shot.
    Fitting cannot start until sync is confirmed via lock.

    Attributes:
        shot_id: Unique identifier for the shot
        offset: Confirmed frame offset
        confidence: Confidence at time of lock
        lock_status: Current lock status
        hdr_start_frame: Start frame in HDR source
        hdr_end_frame: End frame in HDR source
    """

    shot_id: str = ""
    offset: int = 0
    confidence: float = 0.0
    lock_status: LockStatus = LockStatus.UNLOCKED
    hdr_start_frame: int = 0
    hdr_end_frame: int = 0

    def lock(self, offset: int, confidence: float):
        """Lock this shot with the given offset.

        Args:
            offset: The confirmed frame offset
            confidence: Confidence level at lock time
        """
        self.offset = offset
        self.confidence = confidence
        self.lock_status = LockStatus.LOCKED

    def unlock(self):
        """Unlock this shot for re-synchronization."""
        self.lock_status = LockStatus.UNLOCKED

    def mark_needs_review(self):
        """Flag this shot as needing review."""
        self.lock_status = LockStatus.NEEDS_REVIEW

    @property
    def is_locked(self) -> bool:
        """Check if this shot is locked."""
        return self.lock_status == LockStatus.LOCKED

    def to_dict(self) -> dict:
        """Serialize to dictionary for project save."""
        return {
            "shot_id": self.shot_id,
            "offset": self.offset,
            "confidence": self.confidence,
            "lock_status": self.lock_status.value,
            "hdr_start_frame": self.hdr_start_frame,
            "hdr_end_frame": self.hdr_end_frame,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ShotLock":
        """Deserialize from dictionary."""
        lock_status = LockStatus(data.get("lock_status", "unlocked"))
        return cls(
            shot_id=data.get("shot_id", ""),
            offset=data.get("offset", 0),
            confidence=data.get("confidence", 0.0),
            lock_status=lock_status,
            hdr_start_frame=data.get("hdr_start_frame", 0),
            hdr_end_frame=data.get("hdr_end_frame", 0),
        )
