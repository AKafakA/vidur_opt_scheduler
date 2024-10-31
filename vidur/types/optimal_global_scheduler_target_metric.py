from enum import Enum


class TargetMetric(Enum):
    """
    Target metrics for the scheduler
    """
    MIN_LATENCY = 1
    MAX_AVG_BATCH_SIZE = 2
    MAX_MIN_BATCH_SIZE = 3
    MIN_SCHEDULING_DELAY = 4
    MIN_DECODING_DELAY = 5