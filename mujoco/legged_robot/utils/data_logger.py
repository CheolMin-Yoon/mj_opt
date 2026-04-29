# utils/data_logger.py
class DataLogger:
    """
    이름 붙인 시그널을 매 control step에 append하고,
    plot_helpers에 바로 넘길 수 있는 numpy 배열로 반환.
    """
    def __init__(self):
        self._buf = {}

    def log(self, **kwargs):
        for k, v in kwargs.items():
            self._buf.setdefault(k, []).append(np.asarray(v).copy())

    def get(self, key):
        return np.array(self._buf[key])

    def reset(self):
        self._buf.clear()

    def __getitem__(self, key):
        return self.get(key)