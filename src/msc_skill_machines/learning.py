from msc_skill_machines.primitives import TaskPrimitive
from msc_skill_machines.world_value_functions import TabularActionWorldValueFunction
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
    def qlearn(
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
            for ep_i in tqdm( range(num_episodes), desc=f"qlearn[{task.identifier()}]"):

                state, _ = task.reset()
                continue_episode = True
                logger.debug(f"Episode {ep_i}: state = {state}")

                for measure in measures:
                    measure.begin_trajectory()

                while continue_episode:
                    use_random_action = qfunc.empty_action(state) or randgen.random() < explore_rate
                    action = qfunc.argmax_action(state) if not use_random_action else task.sample_action(randgen)
                    logger.debug(f"Episode {ep_i}: state = {state}, action = {action}")

                    next_state, reward, is_terminal, is_truncated, info = task.step(action)
                    logger.debug(f"Episode {ep_i}: next_state = {next_state}, reward = {reward}")

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

    @staticmethod
    def primitive_wqlearn(
        primitive: TaskPrimitive,
        randgen: npr.Generator,
        measures: ty.Collection[Measure],

        discount_factor: float,
        learning_rate: float,
        explore_rate: float,
        num_episodes: int
    ) -> TabularActionWorldValueFunction:
        wqfunc = TabularActionWorldValueFunction()

        goal_buffer: set[int] = set()
        goal_buffer.add( 0 )

        min_reward = 0

        with logging_redirect_tqdm():
            for ep_i in tqdm(range(num_episodes), desc=f"wqlearn[{primitive.identifier()}]"):

                state, _ = primitive.reset()
                continue_episode = True

                target_goal = randgen.choice(list(goal_buffer))
                initial_cons = randgen.integers(0, 2**len(primitive.env.con_labels))
                state = (state[0], initial_cons)

                logger.debug(f"Episode {ep_i}: initial state = {state}, target_goal = {target_goal}")

                for measure in measures:
                    measure.begin_trajectory()

                while continue_episode:
                    use_random_action = wqfunc.empty_action(state, target_goal) or randgen.random() < explore_rate
                    action = wqfunc.argmax_action(state, target_goal) if not use_random_action else primitive.sample_action(randgen)
                    env_action, term_action = action
                    logger.debug(f"Episode {ep_i}: state = {state}, action = {action}")

                    next_state, reward, primitive_is_terminal, is_truncated, info = primitive.step(action)
                    logger.debug(f"Episode {ep_i}: next_state = {next_state}, reward = {reward}")

                    if term_action:
                        goal_buffer.add(next_state)

                    for update_goal in goal_buffer:
                        update_reward = min_reward if primitive_is_terminal else reward
                        td_update = update_reward \
                            + (1 - primitive_is_terminal) * discount_factor * wqfunc.max_action(state, update_goal) \
                            - wqfunc(state, update_goal, action)
                        wqfunc.set(state, update_goal, action, wqfunc(state, update_goal, action) + learning_rate * td_update)

                    state = next_state

                    transition = TransitionData(state, action, next_state, reward, primitive_is_terminal, is_truncated, info)
                    for measure in measures:
                        measure.record_step(transition)

                    continue_episode = not primitive_is_terminal and not is_truncated
                    state = next_state

                for measure in measures:
                    measure.end_trajectory()

        return wqfunc
