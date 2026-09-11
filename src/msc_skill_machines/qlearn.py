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

class GridworldQLearn:

    @staticmethod
    def primitive_qlearn(
        primitive: GridworldTaskPrimitive,
        measures: list[Measure],

        discount_factor: float,
        learning_rate: float,
        explore_rate: float,

        num_episodes: int = 100,
    ) -> npt.NDArray[np.floating]:

        qfunc = np.zeros( shape=(*primitive.env.observation_space.shape, primitive.env.action_space.n), dtype=np.float32 )

        with logging_redirect_tqdm():
            for _ in tqdm( range(num_episodes), desc=f"primitive-qlearn[{primitive.target_label}]"):

                state, _ = primitive.reset()
                continue_episode = True


                for measure in measures:
                    measure.begin_trajectory()

                while continue_episode:
                    action = np.argmax( qfunc[*state] )
                    if primitive.env.npr.random() < explore_rate or np.all(qfunc[*state] == 0):
                        env_action = primitive.env.action_space.sample()

                    next_state, reward, reached_terminal, truncated, info = primitive.step( int(action) )
                    td_update = reward + (1 - reached_terminal) * discount_factor * np.max( qfunc[*next_state] ) - qfunc[*state, action]
                    qfunc[*state, action] += learning_rate * td_update

                    transition = TransitionData(state, action, next_state, reward, reached_terminal, truncated, info)
                    for measure in measures:
                        measure.record_step(transition)

                    continue_episode = not reached_terminal and not truncated
                    state = next_state

                for measure in measures:
                    measure.end_trajectory()
