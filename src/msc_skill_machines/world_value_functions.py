from abc import abstractmethod
from collections import defaultdict
import typing as ty

from msc_skill_machines.value_functions import State, Action, StateValueFunction, ActionValueFunction, TabularState, TabularAction, TabularStateValueFunction, TabularActionValueFunction

type Goal = ty.Any

class StateWorldValueFunction(ty.Protocol):
    """
    ## State World Value Functions

    $$ \\hat{V}: \\mathcal{S} \\times \\mathcal{G} \\mapsto \\mathbb{R} $$
    """

    def __call__(self, state: State, goal: Goal) -> float:
        return self.evaluate(state, goal)

    @abstractmethod
    def evaluate(self, state: State, goal: Goal) -> float:
        pass

    @abstractmethod
    def set(self, state: State, goal: Goal, value: float):
        pass

    @abstractmethod
    def to_state_value_function(self) -> StateValueFunction:
        pass

class ActionWorldValueFunction(ty.Protocol):
    """
    ## Action World Value Functions

    $$ \\hat{G}: \\mathcal{S} \\times \\mathcal{G} \\times \\mathcal{A} \\mapsto \\mathbb{R} $$
    """

    def __call__(self, state: State, goal: Goal, action: Action) -> float:
        return self.evaluate(state, goal, action)

    def evaluate(self, state: State, goal: Goal, action: Action) -> float:
        pass

    def set(self, state: State, goal: Goal, action: Action, value: float):
        pass

    @abstractmethod
    def max_action(self, state: State, goal: Goal) -> float: pass
    @abstractmethod
    def argmax_action(self, state: State, goal: Goal) -> State: pass

    @abstractmethod
    def to_state_world_value_function(self) -> StateWorldValueFunction: pass
    @abstractmethod
    def to_action_value_function(self) -> ActionValueFunction: pass

type TabularGoal = ty.Hashable

class TabularStateWorldValueFunction(StateWorldValueFunction):
    table: dict[TabularState, dict[TabularGoal, float]] = defaultdict(lambda: defaultdict(float))

    def evaluate(self, state: State, goal: Goal) -> float:
        return self.table[state][goal]

    def update(self, state: State, goal: Goal, value: float):
        self.table[state][goal] = value

    def to_state_value_function(self) -> TabularStateValueFunction:
        vfunc = TabularStateValueFunction()
        for state, goals in self.table.items():
            vfunc.update(state, max(goals.values()))
        return vfunc

class TabularActionWorldValueFunction(ActionWorldValueFunction):
    table: dict[TabularState, dict[TabularGoal, dict[TabularAction, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def evaluate(self, state: State, goal: Goal, action: Action) -> float:
        return self.table[state][goal][action]

    def set(self, state: State, goal: Goal, action: Action, value: float):
        self.table[state][goal][action] = value

    def empty_action(self, state: TabularState, goal: TabularGoal) -> bool: return len(self.table[state][goal]) == 0
    def max_action(self, state: TabularState, goal: TabularGoal) -> float: return max(self.table[state][goal].values()) if not self.empty_action(state, goal) else 0.
    def argmax_action(self, state: TabularState, goal: TabularGoal) -> State: return max(self.table[state][goal], key=self.table[state][goal].get)

    def to_state_world_value_function(self) -> TabularStateWorldValueFunction:
        wvfunc = TabularStateWorldValueFunction()
        for state, goals in self.table.items():
            for goal, actions in goals.items():
                wvfunc.update(state, goal, max(actions.values()))
        return wvfunc

    def to_action_value_function(self) -> TabularActionValueFunction:
        qfunc = TabularActionValueFunction()
        for state, goals in self.table.items():

            best_actions: dict[TabularAction, float] = {}
            for goal, actions in goals.items():
                for action, value in actions.items():
                    if action not in best_actions or value > best_actions[action]:
                        best_actions[action] = value

            for action, value in best_actions.items():
                qfunc.update(state, action, value)

        return qfunc
