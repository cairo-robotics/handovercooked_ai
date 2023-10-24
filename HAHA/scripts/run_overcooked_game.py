from pathlib import Path

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

    # bc, human_proxy = BehavioralCloningTrainer.load_bc_and_human_proxy(args, name=f'bc_{args.layout}')
    # worker = load_agent(Path('agent_models/subtask_worker_bcp/best/agents_dir/agent_0'), args)
    tm = HumanPlayer('human', args)
    # agent = HumanManagerHRL(worker, args)


    agent =  load_agent(Path('agent_models/selfplay/best/agents_dir/agent_0'), args)
    # agent = GreedyHumanModel()
    # tm = load_agent(Path('agent_models/old_SP'), args)

    t_idx = 1 - args.p_idx
    # tm = DummyAgent('random')# load_agent(Path(args.teammate), args)
    # tm.set_idx(t_idx, args.layout, is_hrl=isinstance(tm, HierarchicalRL), tune_subtasks=False)
    # if args.agent == 'human':
    #     agent = args.agent
    # else:
    #     agent = load_agent(Path(args.agent), args)
    #     agent.set_idx(args.p_idx, args.layout, is_hrl=isinstance(agent, HierarchicalRL), tune_subtasks=False)

    layout = "cramped_room_single"

    dc = OvercookedGUI(args, agent=agent, teammate=None, layout_name=layout, p_idx=args.p_idx)
    dc.on_execute()

    # score, record = evaluate_agent(agent, layout, num_games=1, horizon=10)
    # print(score)

    # agent_eval = AgentEvaluator.from_layout_name({"layout_name": layout}, {})  # , {"num_players": 1})
    # agent_eval.evaluate_oai_agent(agent, 10)
