from vidur.prediction.predictor.predictor import Predictor


class DummyPredictor(Predictor):
    def __init__(self, config, port):
        super().__init__(config, port)

    def predict(self, target_request):
        return 1.0
