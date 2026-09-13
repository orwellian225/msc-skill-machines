import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from msc_skill_machines.gridworld_builder import build_primitives, load_gridworld, build_tasks
from msc_skill_machines.qlearn import GridworldQLearning
from msc_skill_machines.measures import CumulativeEpisodeReward
from msc_skill_machines.learning import Tabular

toml_path = sys.argv[1] if len(sys.argv) > 1 else "environments/corridor.toml"

spec, env = load_gridworld(toml_path)
primitives = build_primitives(env, seed=spec.seed)
tasks = build_tasks(env, seed=spec.seed)

print(f"loaded {spec.name} from {spec.source}: grid {env.barrier_mask.shape}, labels {env.all_labels}")

for name, task in tasks.items():
    print(f"  task {name}")

    reward_measure = CumulativeEpisodeReward()
    qfunc = Tabular.task_qlearn(
        task, task.env.npr, [reward_measure],
        discount_factor = 0.9,
        learning_rate = 0.1,
        explore_rate = 0.1,
        num_episodes = 10_000
    )

    out_path = Path("artifacts") / spec.name / f"{task.target_label}_{reward_measure.measure_identifier}_figure.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    reward_measure.to_figure(ax)
    ax.set_title(task.target_label)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"  wrote {out_path}")

    state_values = task.state_value_array(qfunc.to_state_value_function())

    value_out_path = Path("artifacts") / spec.name / f"{task.target_label}_state_value_figure.png"
    vfig, vax = plt.subplots()
    sns.heatmap(state_values, ax=vax, cmap="viridis")
    vax.set_title(f"{task.target_label} state value")
    vfig.savefig(value_out_path)
    plt.close(vfig)

    print(f"  wrote {value_out_path}")

# for name, primitive in primitives.items():
#     print(f"  primitive {name}: reward map {primitive.reward_map().shape}, goals at {primitive.target_goal_mask().sum()} cells")

# for name, primitive in primitives.items():
#     reward_measure = CumulativeEpisodeReward()
#     measures = [ reward_measure ]
#     qfunc = GridworldQLearning.primitive_qlearn(
#         primitive,
#         measures,
#         discount_factor = 0.9,
#         learning_rate = 0.1,
#         explore_rate = 0.1,
#         num_episodes = 1_000_000
#     )

#     out_path = Path("artifacts") / spec.name / f"{primitive.target_label}_{reward_measure.measure_identifier}_figure.png"
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     fig, ax = plt.subplots()
#     reward_measure.to_figure(ax)
#     ax.set_title(primitive.target_label)
#     fig.savefig(out_path)
#     plt.close(fig)

#     print(f"  wrote {out_path}")
