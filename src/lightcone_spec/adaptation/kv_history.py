"""Immutable historical drafter-KV version contract."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KVSegment:
    start_token: int
    end_token: int
    source_version: int

    def __post_init__(self) -> None:
        if self.start_token < 0 or self.end_token <= self.start_token:
            raise ValueError("KV segment must be a non-empty half-open interval")
        if self.source_version < 0:
            raise ValueError("source_version must be non-negative")


@dataclass
class FrozenKVHistory:
    """Append-only metadata for KV values that are never reconstructed."""

    segments: list[KVSegment] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.segments[-1].end_token if self.segments else 0

    def append(self, token_count: int, source_version: int) -> KVSegment:
        if token_count < 1:
            raise ValueError("token_count must be positive")
        start = self.length
        segment = KVSegment(start, start + token_count, source_version)
        if (
            self.segments
            and self.segments[-1].source_version == source_version
        ):
            previous = self.segments[-1]
            segment = KVSegment(
                previous.start_token,
                segment.end_token,
                source_version,
            )
            self.segments[-1] = segment
        else:
            self.segments.append(segment)
        return segment

    def retract(self, new_length: int) -> None:
        if new_length < 0 or new_length > self.length:
            raise ValueError("retraction must remain inside the KV history")
        retained: list[KVSegment] = []
        for segment in self.segments:
            if segment.start_token >= new_length:
                break
            retained.append(
                KVSegment(
                    segment.start_token,
                    min(segment.end_token, new_length),
                    segment.source_version,
                )
            )
        self.segments = retained

    def version_at(self, token_index: int) -> int:
        for segment in self.segments:
            if segment.start_token <= token_index < segment.end_token:
                return segment.source_version
        raise IndexError(token_index)
