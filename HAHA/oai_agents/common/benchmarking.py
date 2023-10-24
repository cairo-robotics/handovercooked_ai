import json

import numpy as np

from common.arguments import get_arguments
from gym_environments.base_overcooked_env import OvercookedGymEnv


def evaluate_agent(agent, layout_name, num_games=5, horizon=400):
    args = get_arguments()

    env = OvercookedGymEnv(layout_name=layout_name, args=args, ret_completed_subtasks=True,
                           is_eval_env=True, horizon=horizon)

    agent.set_idx(0, env)
    agent.set_encoding_params(env.mdp, horizon)

    log = np.empty((num_games, horizon, 4), dtype=object)

    for g in range(num_games):
        env.reset()
        agent.reset()
        score = 0
        step = 0

        done = False
        while not done:
            state = env.state

            obs = env.get_obs(env.p_idx, on_reset=False)
            action = agent.predict(obs, state=env.state, deterministic=False)[0]

            obs, reward, done, info = env.step(action)

            curr_reward = sum(info['sparse_r_by_agent'])
            score += curr_reward

            log[g, step, :] = [json.dumps(state.to_dict()), action, curr_reward, score]
            step += 1

    mean_score = np.mean(log[:, -1, 3])
    return mean_score, log





