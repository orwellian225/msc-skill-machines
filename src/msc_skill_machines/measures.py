from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from re import compile

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import typing as ty

@dataclass
class TransitionData:
    state: ty.Any
    action: ty.Any
    next_state: ty.Any
    observed_reward: float
    is_terminal: bool
    is_truncated: bool
    info: dict[str, ty.Any]

measure_pattern = compile('^[a-z0-9]+(-[a-z0-9]+)*$')

class Measure(ABC):
    measure_identifier: str
    user_label: ty.Optional[str]

    def __post_init__(self) -> None:
        if self.user_label is not None:
            self.user_label = self.user_label.lower()
        if self.measure_identifier is not None and measure_pattern.match(self.measure_identifier) is None:
            raise ValueError(f'Invalid measure identifier: {self.measure_identifier}\n\tMust be in lower-kebab-case')

    @abstractmethod
    def record_step(self, transition: TransitionData) -> None:
        ...

    @abstractmethod
    def begin_trajectory(self) -> None:
        ...

    @abstractmethod
    def end_trajectory(self) -> None:
        ...

    @abstractmethod
    def values(self) -> ty.Any:
        ...

    @abstractmethod
    def summary(self) -> dict[str, ty.Any]:
        ...

    @abstractmethod
    def to_figure(self, ax: plt.Axes):
        ...

    def identifier(self, use_short_id: bool = False) -> str:
        return self._short_identifier() if use_short_id else self._long_identifier()

    def _short_identifier(self) -> str:
        return "".join( piece[0] for piece in self.measure_identifier.split('-') ) + (f'[{self.user_label}]' if self.user_label is not None else '')

    def _long_identifier(self) -> str:
        return self.measure_identifier + (f'[{self.user_label}]' if self.user_label is not None else '')

@dataclass
class CumulativeEpisodeReward(Measure):
    rewards: list[float] = field(default_factory=list, init=False)
    measure_identifier: str = field(default='cumulative-episode-reward', init=True)
    user_label: ty.Optional[str] = field(default=None, init=True)

    def begin_trajectory(self) -> None:
        self.rewards.append(0.0)

    def record_step(self, transition: TransitionData) -> None:
        self.rewards[-1] += transition.observed_reward

    def end_trajectory(self) -> None:
        pass

    def values(self) -> list[float]:
        return self.rewards

    def summary(self) -> dict[str, ty.Any]:
        return {
            "mean": np.mean(self.rewards),
            "std": np.std(self.rewards),
            "min": np.min(self.rewards),
            "max": np.max(self.rewards),
        }

    def to_figure(self, ax: plt.Axes) -> None:
        sns.lineplot(
            x=range(len(self.rewards)), y=self.rewards, ax=ax
        )
