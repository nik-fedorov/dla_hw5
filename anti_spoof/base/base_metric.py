class BaseMetric:
    def __init__(self, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__

    def accumulate(self, **batch):
        raise NotImplementedError()

    def result(self):
        raise NotImplementedError()
