from abc import abstractmethod
from typing_extensions import Protocol
from collections import defaultdict
import typing as ty

type State = ty.Any
type Action = ty.Any

class StateValueFunction(Protocol):
    """
        ## State Value Functions

        $$ V: \\mathcal{S} \\mapsto \\mathbb{R} $$
    """

    def __call__(self, state: State) -> float:
        return self.evaluate(state)

    @abstractmethod
    def evaluate(self, state: State) -> float:
        pass

    @abstractmethod
    def update(self, state: State, value: float):
        pass

    @abstractmethod
    def max_state(self) -> float: pass
    @abstractmethod
    def argmax_state(self) -> State: pass

class ActionValueFunction(Protocol):
    """
        ## Action Value Functions

        $$ Q: \\mathcal{S} \\times \\mathcal{A} \\mapsto \\mathbb{R} $$
    """

    def __call__(self, state: State, action: Action) -> float:
        return self.evaluate(state, action)

    @abstractmethod
    def evaluate(self, state: State, action: Action) -> float:
        pass

    @abstractmethod
    def update(self, state: State, action: Action, value: float):
        pass

    @abstractmethod
    def max_action(self, state: State) -> float: pass
    @abstractmethod
    def argmax_action(self, state: State) -> State: pass

    @abstractmethod
    def to_state_value_function(self) -> StateValueFunction: pass


type TabularState = ty.Hashable
type TabularAction = ty.Hashable

class TabularStateValueFunction(StateValueFunction):
    table: dict[TabularState, float]

    def __init__(self):
        self.table = defaultdict(float)

    def evaluate(self, state: State) -> float:
        return self.table[state]

    def update(self, state: State, value: float):
        self.table[state] = value

    def max_state(self) -> float: return max(self.table.values())
    def argmax_state(self) -> TabularState: return max(self.table, key=self.table.get)

class TabularActionValueFunction(ActionValueFunction):
    table: dict[TabularState, dict[TabularAction, float]]

    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(float))

    def evaluate(self, state: TabularState, action: TabularAction) -> float:
        return self.table[state][action]

    def update(self, state: TabularState, action: TabularAction, value: float):
        self.table[state][action] = value

    def empty_action(self, state: TabularState) -> bool: return len(self.table[state]) == 0
    def max_action(self, state: TabularState) -> float: return max(self.table[state].values()) if not self.empty_action(state) else 0.
    def argmax_action(self, state: TabularState) -> TabularAction: return max(self.table[state], key=self.table[state].get)

    def to_state_value_function(self) -> TabularStateValueFunction:
        vfunc = TabularStateValueFunction()
        for state, actions in self.table.items():
            vfunc.update(state, max(actions.values()))
        return vfunc
