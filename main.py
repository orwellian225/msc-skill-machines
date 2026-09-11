import sys

from msc_skill_machines.gridworld_builder import build_primitives, load_gridworld

toml_path = sys.argv[1] if len(sys.argv) > 1 else "environments/office.toml"

spec, env = load_gridworld(toml_path)
primitives = build_primitives(env, seed=spec.seed)

print(f"loaded {spec.name} from {spec.source}: grid {env.barrier_mask.shape}, labels {env.all_labels}")
for name, primitive in primitives.items():
    print(f"  primitive {name}: reward map {primitive.reward_map().shape}, goals at {primitive.target_goal_mask().sum()} cells")
