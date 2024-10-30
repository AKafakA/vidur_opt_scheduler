from vidur.request_latency_predictor.base_request_timeline_predictor import BaseRequestTimelinePredictor


class SimulateRequestDelayPredictor(BaseRequestTimelinePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_scheduling_delay(self, replica_scheduler, request):
        return 0

    def predict_request_makespan(self, replica_scheduler, request):
        return 0

    def predict_average_batch_size(self, replica_scheduler, request):
        return 0

    def predict_average_decoding_latency(self, replica_scheduler, request):
        return 0
