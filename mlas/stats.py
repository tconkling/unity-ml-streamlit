from queue import Queue
from typing import Dict, NamedTuple

from mlagents.trainers.stats import StatsWriter, StatsSummary


class StatsEntry(NamedTuple):
    category: str
    values: Dict[str, StatsSummary]
    step: int


class StatsQueueWriter(StatsWriter):
    """Write stats to a thread-safe queue."""
    def __init__(self, queue: Queue):
        self.queue = queue

    def write_stats(
        self, category: str, values: Dict[str, StatsSummary], step: int
    ) -> None:
        self.queue.put(StatsEntry(category, values, step))
