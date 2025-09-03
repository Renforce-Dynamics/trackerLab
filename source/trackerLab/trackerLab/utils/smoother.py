import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseSmoother(ABC):
    @abstractmethod
    def smooth(self, data: torch.Tensor) -> torch.Tensor:
        pass


class MovingAverageSmoother(BaseSmoother):
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.kernel = torch.ones(1, 1, window_size) / window_size

    def smooth(self, data: torch.Tensor) -> torch.Tensor:
        B, T, D = data.shape
        x = data.permute(0, 2, 1)
        kernel = self.kernel.to(x.device).expand(D, 1, -1)
        x = F.pad(x, (self.window_size - 1, 0), mode='replicate')
        y = F.conv1d(x, kernel, groups=D)
        return y.permute(0, 2, 1)


class ExponentialMovingAverageSmoother(BaseSmoother):
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def smooth(self, data: torch.Tensor) -> torch.Tensor:
        B, T, D = data.shape
        output = torch.zeros_like(data)
        output[:, 0] = data[:, 0]
        alpha = self.alpha
        for t in range(1, T):
            output[:, t] = alpha * data[:, t] + (1 - alpha) * output[:, t - 1]
        return output


class GaussianSmoother(BaseSmoother):
    def __init__(self, sigma: float = 2.0, kernel_size: int = None):
        self.sigma = sigma
        if kernel_size is None:
            kernel_size = int(6 * sigma + 1)
        self.kernel_size = kernel_size
        half = kernel_size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        self.kernel = kernel.view(1, 1, -1)

    def smooth(self, data: torch.Tensor) -> torch.Tensor:
        B, T, D = data.shape
        x = data.permute(0, 2, 1)
        kernel = self.kernel.to(x.device).expand(D, 1, -1)
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad), mode='replicate')
        y = F.conv1d(x, kernel, groups=D)
        return y.permute(0, 2, 1)

class ThresholdFilterSmoother(BaseSmoother):
    def __init__(self, threshold: float = 1e-2):
        self.threshold = threshold

    def smooth(self, data: torch.Tensor) -> torch.Tensor:
        magnitudes = torch.norm(data, dim=-1, keepdim=True)  # shape: [B, T, 1]
        mask = (magnitudes >= self.threshold).float()
        return data * mask

class DeltaFilterSmoother(BaseSmoother):
    def __init__(self, delta_threshold: float = 0.1):
        self.delta_threshold = delta_threshold

    def smooth(self, data: torch.Tensor) -> torch.Tensor:
        B, T, D = data.shape
        output = data.clone()
        delta = data[:, 1:] - data[:, :-1]  # [B, T-1, D]
        delta_magnitude = torch.norm(delta, dim=-1)  # [B, T-1]
        mask = delta_magnitude > self.delta_threshold  # [B, T-1]
        for t in range(1, T):
            output[:, t][mask[:, t - 1]] = output[:, t - 1][mask[:, t - 1]]
        return output

class InterpolatedDeltaFilterSmoother(BaseSmoother):
    def __init__(self, delta_threshold: float = 0.1):
        self.delta_threshold = delta_threshold

    def smooth(self, data: torch.Tensor) -> torch.Tensor:
        # data: [B, T, D]
        B, T, D = data.shape
        output = data.clone()
        delta = data[:, 1:] - data[:, :-1]  # [B, T-1, D]
        delta_magnitude = torch.norm(delta, dim=-1)  # [B, T-1]
        mask = delta_magnitude > self.delta_threshold  # [B, T-1]
        for t in range(1, T - 1):
            mask_t = mask[:, t]
            if mask_t.any():
                interp = 0.5 * (output[:, t - 1] + output[:, t + 1])
                output[:, t][mask_t] = interp[mask_t]
        return output

class PipelineSmoother(BaseSmoother):
    def __init__(self, *smoothers: BaseSmoother):
        self.smoothers = smoothers

    def smooth(self, data: torch.Tensor) -> torch.Tensor:
        for smoother in self.smoothers:
            data = smoother.smooth(data)
        return data

smoother_types = {
    "moving_average": MovingAverageSmoother,
    "exponential_moving_average": ExponentialMovingAverageSmoother,
    "gaussian": GaussianSmoother,
    "threshold_filter": ThresholdFilterSmoother,
    "delta_filter": DeltaFilterSmoother,
    "interpolated_delta_filter": InterpolatedDeltaFilterSmoother,
}

def smoother_create(smoother_type: str, **kwargs) -> BaseSmoother:
    if smoother_type in smoother_types:
        smoother_class = smoother_types[smoother_type]
        return smoother_class(**kwargs)
    elif smoother_type == "pipeline":
        smoothers = [smoother_create(name, **params) for name, params in kwargs.items()]
        return PipelineSmoother(*smoothers)
    elif smoother_type == "none":
        return None
    else:
        raise ValueError(f"Unknown smoother type: {smoother_type}")