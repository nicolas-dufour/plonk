import torch


class SigmoidScheduler:
    def __init__(self, start=-3, end=3, tau=1, clip_min=1e-9):
        self.start = start
        self.end = end
        self.tau = tau
        self.clip_min = clip_min

        self.v_start = torch.sigmoid(torch.tensor(self.start / self.tau))
        self.v_end = torch.sigmoid(torch.tensor(self.end / self.tau))

    def __call__(self, t):
        output = (
            -torch.sigmoid((t * (self.end - self.start) + self.start) / self.tau)
            + self.v_end
        ) / (self.v_end - self.v_start)
        return torch.clamp(output, min=self.clip_min, max=1.0)

    def derivative(self, t):
        x = (t * (self.end - self.start) + self.start) / self.tau
        sigmoid_x = torch.sigmoid(x)
        # Chain rule: d/dt of original function
        return (
            -(self.end - self.start)
            * sigmoid_x
            * (1 - sigmoid_x)
            / (self.tau * (self.v_end - self.v_start))
        )

    def alpha(self, t):
        return -self.derivative(t) / (1e-6 + self.__call__(t))


class LinearScheduler:
    def __init__(self, start=1, end=0, clip_min=1e-9):
        self.start = start
        self.end = end
        self.clip_min = clip_min

    def __call__(self, t):
        output = (self.end - self.start) * t + self.start
        return torch.clamp(output, min=self.clip_min, max=1.0)

    def derivative(self, t):
        return torch.tensor(self.end - self.start).to(t.device)

    def alpha(self, t):
        return -self.derivative(t) / (1e-6 + self.__call__(t))


class CosineScheduler:
    def __init__(
        self,
        start: float = 1,
        end: float = 0,
        tau: float = 1.0,
        clip_min: float = 1e-9,
    ):
        self.start = start
        self.end = end
        self.tau = tau
        self.clip_min = clip_min

        self.v_start = torch.cos(torch.tensor(self.start) * torch.pi / 2) ** (
            2 * self.tau
        )
        self.v_end = torch.cos(torch.tensor(self.end) * torch.pi / 2) ** (2 * self.tau)

    def __call__(self, t: float) -> float:
        output = (
            torch.cos((t * (self.end - self.start) + self.start) * torch.pi / 2)
            ** (2 * self.tau)
            - self.v_end
        ) / (self.v_start - self.v_end)
        return torch.clamp(output, min=self.clip_min, max=1.0)

    def derivative(self, t: float) -> float:
        x = (t * (self.end - self.start) + self.start) * torch.pi / 2
        cos_x = torch.cos(x)
        # Chain rule: d/dt of original function
        return (
            -2
            * self.tau
            * (self.end - self.start)
            * torch.pi
            / 2
            * cos_x
            * (cos_x ** (2 * self.tau - 1))
            * torch.sin(x)
            / (self.v_start - self.v_end)
        )


class CosineSchedulerSimple:
    def __init__(self, ns: float = 0.0002, ds: float = 0.00025):
        self.ns = ns
        self.ds = ds

    def __call__(self, t: float) -> float:
        return torch.cos(((t + self.ns) / (1 + self.ds)) * torch.pi / 2) ** 2

    def derivative(self, t: float) -> float:
        x = ((t + self.ns) / (1 + self.ds)) * torch.pi / 2
        return -torch.pi * torch.cos(x) * torch.sin(x) / (1 + self.ds)
