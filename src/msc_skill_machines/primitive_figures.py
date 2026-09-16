import matplotlib.pyplot as plt
import seaborn as sns

from msc_skill_machines.primitives import GridworldTaskPrimitive
from msc_skill_machines.value_functions import ActionValueFunction


def primitive_action_value_figure(
    primitive: GridworldTaskPrimitive,
    qfunc: ActionValueFunction,
    cmap: str = "viridis",
) -> plt.Figure:
    """Render a primitive's action-value function as a grid of heatmaps.

    One column per env action, two rows: no-terminate on top, terminate below::

        a0 no-term   a1 no-term   a2 no-term   a3 no-term
        a0 term      a1 term      a2 term      a3 term

    Matches the layer layout of :meth:`GridworldTaskPrimitive.env_action_vfunc`.
    """
    values = primitive.env_action_vfunc(qfunc)
    n_actions = len(primitive.env.transitions)
    vmin, vmax = float(values.min()), float(values.max())

    fig, axes = plt.subplots(2, n_actions, figsize=(4 * n_actions, 8), squeeze=False)
    for term in range(2):
        for action in range(n_actions):
            layer = term * n_actions + action
            sns.heatmap(
                values[:, :, layer], ax=axes[term][action], cmap=cmap,
                vmin=vmin, vmax=vmax, cbar=action == n_actions - 1,
            )
            axes[term][action].set_title(f"action {action}{' term' if term else ' no-term'}")

    fig.suptitle(primitive.target_label)
    fig.tight_layout()
    return fig
