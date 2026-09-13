from operator import is_
import logging

import numpy as np
import numpy.random as npr

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from msc_skill_machines.measures import Measure, TransitionData
from msc_skill_machines.value_functions import TabularActionValueFunction

import typing as ty

logger = logging.getLogger(__name__)

class Tabular:

    @staticmethod
    def task_qlearn(
        task,
        randgen: npr.Generator,
        measures: ty.Collection[Measure],

        discount_factor: float,
        learning_rate: float,
        explore_rate: float,
        num_episodes: int

    ) -> TabularActionValueFunction:

        qfunc = TabularActionValueFunction()

        with logging_redirect_tqdm():
            for _ in tqdm( range(num_episodes), desc=f"task-qlearn[{task.identifier()}]"):

                state, _ = task.reset()
                continue_episode = True

                for measure in measures:
                    measure.begin_trajectory()

                while continue_episode:
                    use_random_action = qfunc.empty_action(state) or randgen.random() < explore_rate
                    action = qfunc.argmax_action(state) if not use_random_action else task.sample_action(randgen)

                    next_state, reward, is_terminal, is_truncated, info = task.step(action)
                    td_update = reward + (1 - is_terminal) * discount_factor * qfunc.max_action(next_state) - qfunc(state, action)
                    qfunc.update(state, action, qfunc(state, action) + learning_rate * td_update)

                    transition = TransitionData(state, action, next_state, reward, is_terminal, is_truncated, info)
                    for measure in measures:
                        measure.record_step(transition)

                    continue_episode = not is_terminal and not is_truncated
                    state = next_state

                for measure in measures:
                    measure.end_trajectory()

        return qfunc
