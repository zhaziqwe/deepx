class Optimizer:
    def __init__(self, params,defaults: dict[str, Any]) -> None:
        self.params = params
        self.defaults = defaults

    def step(self):
        pass