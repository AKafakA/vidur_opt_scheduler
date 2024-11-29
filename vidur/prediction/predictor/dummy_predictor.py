from vidur.prediction.predictor.predictor import Predictor


class DummyPredictor(Predictor):
    def __init__(self, config, port):
        super().__init__(config, port)

    def predict(self, target_request, target_metric):
        return 1.0

    def update(self, request):
        pass

    def reset(self):
        pass
