from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments


# SP
def get_selfplay_agent(args, training_steps=1e7, tag=None):
    name = 'selfplay'
    try:
        tag = tag or 'best'
        agents = RLAgentTrainer.load_agents(args, name=name, tag=tag)
    except FileNotFoundError as e:
        print(f'Could not find saved selfplay agent, creating them from scratch...\nFull Error: {e}')
        selfplay_trainer = RLAgentTrainer([], args, selfplay=True, name=name, seed=678, use_frame_stack=False,
                                          use_lstm=False, use_cnn=False)
        selfplay_trainer.train_agents(train_timesteps=training_steps)
        agents = selfplay_trainer.get_agents()
    return agents


if __name__ == '__main__':
    args = get_arguments()
    get_selfplay_agent(args, training_steps=1e7, tag="a2")
    print('GOT SP', flush=True)
