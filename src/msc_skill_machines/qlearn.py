from collections import defaultdict
import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import typing as ty
import numpy.typing as npt

from msc_skill_machines.environments import GridworldEnv
from msc_skill_machines.primitives import GridworldTaskPrimitive
from msc_skill_machines.measures import Measure, TransitionData

logger = logging.getLogger(__name__)

class GridworldQLearning:

    @staticmethod
    def primitive_qlearn(
        primitive: GridworldTaskPrimitive,
        measures: list[Measure],

        discount_factor: float,
        learning_rate: float,
        explore_rate: float,

        num_episodes: int = 100,
    ) -> dict[ty.Any, dict[ty.Any, float]]:
        qfunc = defaultdict(lambda: defaultdict(lambda: 0.0))

        def state_key(state):
            return (
                tuple(state[0]),
                int(state[1] @ (1 << np.arange(state[1].shape[0]))),
            )
        def action_key(action):
            return action[0] + primitive.env.action_space.n * int(action[1])

        with logging_redirect_tqdm():
            for _ in tqdm( range(num_episodes), desc=f"primitive-qlearn[{primitive.target_label}]"):

                state, _ = primitive.reset()
                continue_episode = True

                for measure in measures:
                    measure.begin_trajectory()

                while continue_episode:
                    if len(qfunc[state_key(state)]) == 0 or primitive.env.npr.random() < explore_rate:
                        env_action = primitive.env.action_space.sample()
                        ter_action = primitive.env.npr.random() > 0.5
                    else:
                        best_action_key = max(qfunc[state_key(state)], key=qfunc[state_key(state)].get)
                        env_action = best_action_key % primitive.env.action_space.n
                        ter_action = bool(best_action_key // primitive.env.action_space.n)
                    action = (env_action, ter_action)

                    next_state, reward, reached_terminal, truncated, info = primitive.step( ( int(env_action), ter_action ) )
                    next_state_values = qfunc[state_key(next_state)].values()
                    best_next_value = max(next_state_values) if next_state_values else 0.0
                    td_update = reward + (1 - reached_terminal) * discount_factor * best_next_value - qfunc[state_key(state)][action_key(action)]

                    qfunc[state_key(state)][action_key(action)] += learning_rate * td_update

                    transition = TransitionData(state, action, next_state, reward, reached_terminal, truncated, info)
                    for measure in measures:
                        measure.record_step(transition)

                    continue_episode = not reached_terminal and not truncated
                    state = next_state

                for measure in measures:
                    measure.end_trajectory()

        return qfunc
