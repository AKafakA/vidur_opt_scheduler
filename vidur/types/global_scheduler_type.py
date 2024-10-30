from vidur.types.base_int_enum import BaseIntEnum


class GlobalSchedulerType(BaseIntEnum):
    RANDOM = 1
    ROUND_ROBIN = 2
    LOR = 3
    LODT = 4
    OPT_LATENCY = 5
    OPT_THROUGHPUT = 6
    OPT_SCHEDULING_DELAY = 7
    OPT_DECODING_DELAY = 8