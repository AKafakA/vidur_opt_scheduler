from enum import Enum


class TargetMetric(Enum):
    """
    Target metrics for the scheduler
    """
    LATENCY = 1
    THROUGHPUT = 2
    SCHEDULING_DELAY = 3
    DECODING_DELAY = 4