from collections import OrderedDict
from syn10_diffusion import utils

utils.seed_all()


class EMA:
    _shadow_exists = False

    def __init__(self, ddp_model, decay, delay):
        self.ddp_model = ddp_model
        self.decay = decay
        self.delay = delay
        self.shadow = OrderedDict()

    def build_shadow(self):
        if not self._shadow_exists:
            for name, param in self.ddp_model.module.named_parameters():
                if param.requires_grad:
                    if name not in self.shadow:
                        self.shadow[name] = param.data.detach().clone()
            self._shadow_exists = True

    def step(self):
        if self._shadow_exists:
            for name, param in self.ddp_model.module.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    self.shadow[name].mul_(self.decay).add_(param.data.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        if not self._shadow_exists:
            self.shadow = state_dict
            self._shadow_exists = True
