from vidur.request_timeline_predictor.request_timeline_predictor_registry import RequestTimelinePredictorRegistry
from vidur.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler
from vidur.scheduler.global_scheduler.random_global_scheduler import (
    RandomGlobalScheduler,
)
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import (
    RoundRobinGlobalScheduler,
)
from vidur.scheduler.global_scheduler.lodt_scheduler import LODTScheduler
from vidur.scheduler.global_scheduler.length_aware_optimal_scheduler import LengthAwareOptimalScheduler
from vidur.types import GlobalSchedulerType
from vidur.utils.base_registry import BaseRegistry


class GlobalSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> GlobalSchedulerType:
        return GlobalSchedulerType.from_str(key_str)


GlobalSchedulerRegistry.register(GlobalSchedulerType.RANDOM, RandomGlobalScheduler)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.ROUND_ROBIN, RoundRobinGlobalScheduler
)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR, LORGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LODT, LODTScheduler)

GlobalSchedulerRegistry.register(GlobalSchedulerType.OPT, LengthAwareOptimalScheduler)
# GlobalSchedulerRegistry.register(GlobalSchedulerType.OPT_LATENCY, LengthAwareOptimalScheduler(
#     target_metric=TargetMetric.MIN_LATENCY, request_timeline_predictor=request_timeline_predictor))
# GlobalSchedulerRegistry.register(GlobalSchedulerType.OPT_SCHEDULING_DELAY, LengthAwareOptimalScheduler(
#     target_metric=TargetMetric.MIN_SCHEDULING_DELAY, request_timeline_predictor=request_timeline_predictor))
# GlobalSchedulerRegistry.register(GlobalSchedulerType.OPT_DECODING_DELAY, LengthAwareOptimalScheduler(
#     target_metric=TargetMetric.MIN_DECODING_DELAY, request_timeline_predictor=request_timeline_predictor))
# GlobalSchedulerRegistry.register(GlobalSchedulerType.OPT_MAX_AVG_BATCH_SIZE, LengthAwareOptimalScheduler(
#     target_metric=TargetMetric.MAX_AVG_BATCH_SIZE, request_timeline_predictor=request_timeline_predictor))
# GlobalSchedulerRegistry.register(GlobalSchedulerType.OPT_MAX_MIN_BATCH_SIZE, LengthAwareOptimalScheduler(
#     target_metric=TargetMetric.MAX_MIN_BATCH_SIZE, request_timeline_predictor=request_timeline_predictor))

