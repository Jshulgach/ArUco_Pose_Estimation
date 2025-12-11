import numpy as np

class PredictivePoseFilter:
    def __init__(self, alpha=0.5):
        self.prev_rvec = None
        self.prev_tvec = None
        self.velocity_r = None
        self.velocity_t = None
        self.alpha = alpha

    def update(self, rvec, tvec):
        if self.prev_rvec is not None and rvec is not None:
            self.velocity_r = (rvec - self.prev_rvec)
            self.velocity_t = (tvec - self.prev_tvec)

        self.prev_rvec = rvec
        self.prev_tvec = tvec

    def predict(self):
        if self.prev_rvec is not None and self.velocity_r is not None:
            rvec_pred = self.prev_rvec + self.alpha * self.velocity_r
            tvec_pred = self.prev_tvec + self.alpha * self.velocity_t
            return rvec_pred, tvec_pred
        return None, None

    def reset(self):
        self.prev_rvec = None
        self.prev_tvec = None
        self.velocity_r = None
        self.velocity_t = None