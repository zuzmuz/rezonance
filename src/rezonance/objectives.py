from numpy import argmax
import torch
from torch import Tensor, nn


class Metric:
    def __repr__(self) -> str: ...

    def __iadd__(self, rhs) -> Metric: ...

    def __add__(self, rhs) -> Metric: ...

    def __truediv__(self, rhs) -> Metric: ...

    def __itruediv__(self, rhs) -> Metric: ...


class LossMetric(Metric):
    def __init__(self, loss: torch.types.Number = 0):
        self.loss = loss

    def __repr__(self) -> str:
        return f"Loss = {self.loss}"

    def __add__(self, rhs) -> Metric:
        return LossMetric(self.loss + rhs.loss)

    def __iadd__(self, rhs) -> Metric:
        self.loss += rhs.loss
        return self

    def __truediv__(self, rhs) -> Metric:
        return LossMetric(self.loss / rhs)

    def __itruediv__(self, rhs) -> Metric:
        self.loss /= rhs
        return self


class ClassificationMetric(Metric):
    def __init__(
        self,
        loss: torch.types.Number = 0,
        accuracy: torch.types.Number = 0,
    ):
        self.loss = loss
        self.accuracy = accuracy

    def __repr__(self) -> str:
        return f"Loss = {self.loss}, Accuracy = {self.accuracy}"

    def __add__(self, rhs) -> Metric:
        return ClassificationMetric(
            self.loss + rhs.loss, self.accuracy + rhs.accuracy
        )

    def __iadd__(self, rhs) -> Metric:
        self.loss += rhs.loss
        self.accuracy += rhs.accuracy
        return self

    def __truediv__(self, rhs) -> Metric:
        return ClassificationMetric(
            self.loss / rhs, self.accuracy / rhs
        )

    def __itruediv__(self, rhs) -> Metric:
        self.loss /= rhs
        self.accuracy /= rhs
        return self


# TODO this a very important class, find better name
class Objective:
    def output_size(self) -> int: ...

    def forward(self, pitch: Tensor) -> Tensor: ...

    def backward(self, output: Tensor) -> Tensor: ...

    def loss(
        self, predictions: Tensor, labels: Tensor, log: bool = False
    ) -> tuple[Tensor, Metric | None]: ...
    
    def get_metric(self) -> Metric: ...


class BasicObjective(Objective):
    def __init__(self):
        self.criterion = nn.MSELoss()

    def output_size(self) -> int:
        return 1

    def forward(self, pitch: Tensor) -> Tensor:
        return pitch

    def loss(
        self, predictions: Tensor, labels: Tensor, log: bool = False
    ) -> tuple[Tensor, Metric | None]:
        loss = self.criterion(predictions, labels)
        if log:
            return loss, LossMetric(loss.item())
        return loss, None
    
    def get_metric(self) -> Metric:
        return LossMetric()

class CyclicPitchObjective(Objective):
    def __init__(self, with_octave: bool):
        self.with_octave = with_octave
        self.criterion = nn.MSELoss()

    def output_size(self) -> int:
        return 3 if self.with_octave else 2

    def forward(self, pitch: Tensor) -> Tensor:
        return (
            torch.cat(
                [
                    torch.sin(pitch * torch.pi / 6),
                    torch.cos(pitch * torch.pi / 6),
                    pitch / 12 - 1,
                ],
                dim=-1,
            )
            if self.with_octave
            else torch.cat(
                [
                    torch.sin(pitch * torch.pi / 6),
                    torch.cos(pitch * torch.pi / 6),
                ],
                dim=-1,
            )
        )

    def loss(
        self, predictions: Tensor, labels: Tensor, log: bool = False
    ) -> tuple[Tensor, Metric | None]:
        loss = self.criterion(predictions, labels)
        if log:
            return loss, LossMetric(loss.item())
        return loss, None

    def get_metric(self) -> Metric:
        return LossMetric()


class NoteClassifierObjective(Objective):
    def __init__(
        self,
        min_pitch: torch.types.Number,
        max_pitch: torch.types.Number,
        pitch_step: torch.types.Number,
    ):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.bins_per_pitch = 1 / pitch_step
        self.criterion = nn.CrossEntropyLoss()

    def output_size(self) -> int:
        return int(
            round(
                (self.max_pitch - self.min_pitch)
                * self.bins_per_pitch
            )
        )

    def _get_pitch_index(self, pitch: Tensor) -> Tensor:
        return (
            ((pitch - self.min_pitch) * self.bins_per_pitch)
            .round()
            .long()
        )

    def forward(self, pitch: Tensor) -> Tensor:
        return self._get_pitch_index(pitch).flatten()

    def loss(
        self, predictions: Tensor, labels: Tensor, log: bool = False
    ) -> tuple[Tensor, Metric | None]:
        loss: Tensor = self.criterion(predictions, labels)

        if log:
            loss_value = loss.item()
            accuracy = (
                (predictions.argmax(-1) == labels).float().mean()
            )
            return loss, ClassificationMetric(loss_value, accuracy)
        return loss, None


    def get_metric(self) -> Metric:
        return ClassificationMetric()
