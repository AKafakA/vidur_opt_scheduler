from vidur.request_latency_predictor.request_timeline_predictor_registry import RequestTimelinePredictorRegistry
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
from vidur.types.optimal_global_scheduler_target_metric import TargetMetric
from vidur.types.request_timeline_predictor_type import RequestTimelinePredictorType
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

request_timeline_predictor = RequestTimelinePredictorRegistry.get(RequestTimelinePredictorType.SIMULATE)
GlobalSchedulerRegistry.register(GlobalSchedulerType.OPT_LATENCY, LengthAwareOptimalScheduler(
    target_metric=TargetMetric.LATENCY, request_timeline_predictor=request_timeline_predictor))
GlobalSchedulerRegistry.register(GlobalSchedulerType.OPT_SCHEDULING_DELAY, LengthAwareOptimalScheduler(
    target_metric=TargetMetric.SCHEDULING_DELAY, request_timeline_predictor=request_timeline_predictor))

