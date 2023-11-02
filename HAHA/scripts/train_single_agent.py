import matplotlib.pyplot as plt
import pandas

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments


# SP
def train_selfplay_agent(args, training_steps=1e7, n_checkpoints=0, save_filename=None):
    name = 'selfplay'

    checkpoint_rate = training_steps // n_checkpoints if n_checkpoints > 0 else None
    selfplay_trainer = RLAgentTrainer([], args, selfplay=True, name=name, seed=678, use_frame_stack=False,
                                      use_lstm=False, use_cnn=False, fcp_ck_rate=checkpoint_rate)
    selfplay_trainer.train_agents(train_timesteps=training_steps, data_filename=save_filename)
    agents = selfplay_trainer.get_agents()

    return agents


def plot_training_data(data_file):
    training_data = pandas.read_csv(data_file)

    plt.plot(training_data['timestep'], training_data['mean_reward'])

    # Format plot
    ax = plt.gca()
    ax.set_xlim([0, training_data['timestep'].iloc[-1]])
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Average Reward')

    ax2 = ax.twiny()
    last_time = training_data['time'].iloc[-1]
    if last_time > 60 * 5:
        last_time /= 60
        ax2.set_xlabel('Time (m)')
    else:
        ax2.set_xlabel('Time (s)')
    ax2.set_xlim([0, last_time])

    plt.show()


if __name__ == '__main__':
    training_steps = 5e6
    filename = f'{training_steps:.0E}-random-recipe-data.csv'

    # --n-envs 9 --layout-names cramped_room_single,cramped_room_single_v2,cramped_room_single_v3,cramped_room_single_v4,cramped_room_single_v5,cramped_room_single_v6,cramped_room_single_v7,cramped_room_single_v8,cramped_room_single_v9,cramped_room_single_v10
    # --layout-names cramped_room_single_recipes
    args = get_arguments()
    train_selfplay_agent(args, training_steps=training_steps, n_checkpoints=10, save_filename=filename)

    plot_training_data(filename)

    print('GOT SP', flush=True)
