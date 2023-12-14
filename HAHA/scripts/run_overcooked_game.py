from pathlib import Path

import numpy as np

from agents.human_proxy import HumanProxy
from common.benchmarking import evaluate_agent
from oai_agents.agents.agent_utils import load_agent
from oai_agents.agents.human_agents import HumanPlayer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import HandoverOvercookedGUI, OvercookedGUI
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Sample commands
    python scripts/run_overcooked_game.py --agent human --teammate agent_models/HAHA
    """
    additional_args = [
        ('--agent', {'type': str, 'default': 'human', 'help': '"human" to used keyboard inputs or a path to a saved agent'}),
        ('--teammate', {'type': str, 'default': 'agent_models/HAHA', 'help': 'Path to saved agent to use as teammate'}),
        ('--layout', {'type': str, 'default': 'counter_circuit_o_1order', 'help': 'Layout to play on'}),
        ('--p-idx', {'type': int, 'default': 0, 'help': 'Player idx of agent (teammate will have other player idx), Can be 0 or 1.'})
    ]


    args = get_arguments(additional_args)

    human = HumanPlayer('human', args)
    proxy = HumanProxy('proxy', args)
    agent = load_agent(Path('agent_models/selfplay/ck_3/agents_dir/agent_0'), args)

    layout = "cramped_room_single"

    # score, record = evaluate_agent(agent, layout, num_games=5, horizon=1200)
    # print(score)

    # dc = OvercookedGUI(args, agent=agent, teammate=None, layout_name=layout, p_idx=args.p_idx, fps=30)
    # dc.on_execute()

    # Simulate and visualize Handover Overcooked run
    agents = [agent]
    dc = HandoverOvercookedGUI(args, ai_agents=agents, teammates=None, layout_name=layout, p_idx=args.p_idx, fps=50,
                               threshold=0.5, display=True, horizon=400, save_name=None)
    dc.on_execute()

    # For evaluating different confidence thresholds
    thresholds = [np.Inf, 1, 0.5, 0.2, 0.1, 0.01, 0.001, np.NINF]
    horizon = 400
    n_games = 50
    rewards = np.zeros((len(thresholds), n_games, horizon+1, len(agents)+1))
    handover_percents = np.zeros((len(thresholds), n_games))
    for i, threshold in enumerate(thresholds):
        for j in range(n_games):
            dc = HandoverOvercookedGUI(args, ai_agents=agents, teammates=None, layout_name=layout, p_idx=args.p_idx,
                                       fps=50, threshold=threshold, display=False, horizon=horizon)
            rewards[i, j, :, :], handover_percents[i, j] = dc.on_execute()

            # plt.plot(rewards.mean(axis=(1, 3))[i, :],
            #          label=f"Threshold: {threshold}, Handover %: {handover_percents[i, j] * 100:.0f}%")

    # plt.legend()

    # Bar graph for mean rewards
    labels = [f"{thresholds[i]:.0e}({np.mean(handover_percents, axis=1)[i]*100:.0f}%)" for i in range(len(thresholds))]
    plt.bar(labels, rewards.mean(axis=(1, 3))[:, -1])
    plt.xlabel("Confidence Threshold (Handover %)")
    plt.ylabel("Mean Combined Reward")
    plt.title(f"Mean Combined Reward (over {n_games} games) vs Confidence Threshold")

    plt.show()
    best_threshold = thresholds[rewards.mean(axis=(1, 3))[:, -1].argmax()]

    dc = HandoverOvercookedGUI(args, ai_agents=agents, teammates=None, layout_name=layout, p_idx=args.p_idx, fps=50,
                               threshold=best_threshold, display=True, horizon=400, save_name=None)
    dc.on_execute()
