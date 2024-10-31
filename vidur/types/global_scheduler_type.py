from vidur.types.base_int_enum import BaseIntEnum


class GlobalSchedulerType(BaseIntEnum):
    RANDOM = 1
    ROUND_ROBIN = 2
    LOR = 3
    LODT = 4
    # opt global scheduler name
    OPT_LATENCY = 5
    OPT_SCHEDULING_DELAY = 6
    OPT_DECODING_DELAY = 7
    OPT_MAX_AVG_BATCH_SIZE = 8
    OPT_MAX_MIN_BATCH_SIZE = 9
