from pathlib import Path

from agents.human_proxy import HumanProxy
from common.benchmarking import evaluate_agent
from oai_agents.agents.agent_utils import DummyAgent, load_agent
from oai_agents.agents.hrl import HierarchicalRL
from oai_agents.agents.il import BehavioralCloningTrainer
from oai_agents.agents.human_agents import HumanManagerHRL, HumanPlayer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import OvercookedGUI
from overcooked_ai_py.agents.agent import AgentGroup, GreedyHumanModel
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

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
    agent = load_agent(Path('agent_models/selfplay/handover/agents_dir/agent_0'), args)

    layout = "cramped_room_single_recipe1"

    score, record = evaluate_agent(agent, layout, num_games=1, horizon=400)
    print(score)

    dc = OvercookedGUI(args, agent=proxy, teammate=None, layout_name=layout, p_idx=args.p_idx)
    dc.on_execute()
