from vidur.types.base_int_enum import BaseIntEnum


class RequestTimelinePredictorType(BaseIntEnum):
    DUMMY = 1
    RANDOM_FORREST = 2
    LINEAR_REGRESSION = 3
    SIMULATE = 4
