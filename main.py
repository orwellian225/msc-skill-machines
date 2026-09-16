import sys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from msc_skill_machines.gridworld_builder import build_primitives, load_gridworld, build_tasks
from msc_skill_machines.qlearn import GridworldQLearning
from msc_skill_machines.measures import CumulativeEpisodeReward
from msc_skill_machines.learning import Tabular
from msc_skill_machines.primitive_figures import primitive_action_value_figure

toml_path = sys.argv[1] if len(sys.argv) > 1 else "environments/office.toml"

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler(),
    ],
)

spec, env = load_gridworld(toml_path)
primitives = build_primitives(env, seed=spec.seed)
tasks = build_tasks(env, seed=spec.seed)

print(f"loaded {spec.name} from {spec.source}: grid {env.barrier_mask.shape}, labels {env.all_labels}")

# for name, task in tasks.items():
#     print(f"  task {name}")

#     reward_measure = CumulativeEpisodeReward()
#     qfunc = Tabular.task_qlearn(
#         task, task.env.npr, [reward_measure],
#         discount_factor = 0.9,
#         learning_rate = 0.1,
#         explore_rate = 0.1,
#         num_episodes = 10_000
#     )

#     out_path = Path("artifacts") / spec.name / f"task_{task.target_label}_{reward_measure.measure_identifier}_figure.png"
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     fig, ax = plt.subplots()
#     reward_measure.to_figure(ax)
#     ax.set_title(task.target_label)
#     fig.savefig(out_path)
#     plt.close(fig)

#     print(f"  wrote {out_path}")

#     state_values = task.state_value_array(qfunc.to_state_value_function())

#     value_out_path = Path("artifacts") / spec.name / f"task_{task.target_label}_state_value_figure.png"
#     vfig, vax = plt.subplots()
#     sns.heatmap(state_values, ax=vax, cmap="viridis")
#     vax.set_title(f"{task.target_label} state value")
#     vfig.savefig(value_out_path)
#     plt.close(vfig)

#     print(f"  wrote {value_out_path}")

for name, primitive in primitives.items():
    print(f"  primitive {name}:")

    reward_measure = CumulativeEpisodeReward()
    measures = [ reward_measure ]
    # qfunc = Tabular.qlearn(
    #     primitive, primitive.env.npr, [reward_measure],
    #     discount_factor = 0.9,
    #     learning_rate = 0.1,
    #     explore_rate = 0.1,
    #     num_episodes = 1_000_000
    # )
    wqfunc = Tabular.primitive_wqlearn(
        primitive, primitive.env.npr, [reward_measure],
        discount_factor = 0.9,
        learning_rate = 0.1,
        explore_rate = 0.1,
        num_episodes = 10
    )

    out_path = Path("artifacts") / spec.name / f"primitive_{primitive.target_label}_{reward_measure.measure_identifier}_figure.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    reward_measure.to_figure(ax)
    ax.set_title(primitive.target_label)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"  wrote {out_path}")

    qvalue_fig = primitive_action_value_figure(primitive, wqfunc.to_action_value_function())
    qvalue_out_path = Path("artifacts") / spec.name / f"primitive_{primitive.target_label}_qvalues_figure.png"
    qvalue_fig.savefig(qvalue_out_path)
    plt.close(qvalue_fig)

    print(f"  wrote {qvalue_out_path}")
