import json
import numpy as np
import pandas as pd
import pygame
import pylsl
from pygame import K_UP, K_LEFT, K_RIGHT, K_DOWN, K_SPACE, K_s
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, FULLSCREEN
import matplotlib
import time

from agents.human_proxy import HumanProxy
from oai_agents.agents.confidence import ConfidenceRecord

matplotlib.use('TkAgg')

from os import listdir, environ, system, name
from os.path import isfile, join
import re
import time

from pathlib import Path
import pathlib

USING_WINDOWS = (name == 'nt')
# Windows path

if USING_WINDOWS:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# Lab streaming layer
from pylsl import StreamInfo, StreamOutlet, local_clock

# Used to activate game window at game start for immediate game play
if USING_WINDOWS:
    import pygetwindow as gw

from oai_agents.agents.agent_utils import DummyPolicy
from oai_agents.agents.base_agent import OAIAgent
from oai_agents.agents.il import BehaviouralCloningAgent
from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.agents.hrl import HierarchicalRL
# from oai_agents.agents import Manager
from oai_agents.common.arguments import get_arguments
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from oai_agents.agents.agent_utils import load_agent, DummyAgent
from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv
from oai_agents.gym_environments.manager_env import OvercookedManagerGymEnv
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action, OvercookedState, OvercookedGridworld
# from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer, roboto_path
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from scripts.train_agents import get_bc_and_human_proxy


class OvercookedGUI:
    """Class to run an Overcooked Gridworld game, leaving one of the agents as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""

    def __init__(self, args, layout_name=None, agent=None, teammate=None, p_idx=0, horizon=400,
                 trial_id=None, user_id=None, stream=None, outlet=None, fps=5):
        self.x = None
        self._running = True
        self._display_surf = None
        self.args = args
        self.layout_name = layout_name or 'asymmetric_advantages'

        self.use_subtask_env = False
        if self.use_subtask_env:
            kwargs = {'single_subtask_id': 10, 'args': args, 'is_eval_env': True}
            self.env = OvercookedSubtaskGymEnv(**p_kwargs, **kwargs)
        else:
            self.env = OvercookedGymEnv(layout_name=self.layout_name, args=args, ret_completed_subtasks=True,
                                        is_eval_env=True, horizon=horizon)
        self.agent = agent
        self.p_idx = p_idx
        if teammate is not None:
            self.env.set_teammate(teammate)
            self.env.teammate.set_idx(self.env.t_idx, self.env, is_hrl=isinstance(self.env.teammate, HierarchicalRL),
                                      tune_subtasks=False)
            self.teammate_name = teammate.name
        else:
            self.teammate_name = "None"

        self.env.reset(p_idx=self.p_idx)
        if self.agent != 'human':
            self.agent.set_idx(self.p_idx, self.env, is_hrl=isinstance(self.agent, HierarchicalRL), tune_subtasks=False)

        self.grid_shape = self.env.grid_shape
        self.trial_id = trial_id
        self.user_id = user_id
        self.fps = fps

        self.score = 0
        self.curr_tick = 1
        self.num_collisions = 0
        self.human_action = None
        self.data_path = args.base_dir / args.data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.tile_size = 50

        self.info_stream = stream
        self.outlet = outlet
        # Currently unused, but keeping in case we need it in the future.
        self.collect_trajectory = False

    def start_screen(self):
        pygame.init()
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state,
                                                                         grid=self.env.env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": 1, "score": 0})

        self.surface_size = surface.get_size()
        self.x, self.y = (1920 - self.surface_size[0]) // 2, (1080 - self.surface_size[1]) // 2
        self.grid_shape = self.env.mdp.shape
        self.hud_size = self.surface_size[1] - (self.grid_shape[1] * self.tile_size)
        environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.x, self.y)

        self.window = pygame.display.set_mode(self.surface_size, HWSURFACE | DOUBLEBUF | RESIZABLE)

        pygame.font.init()
        start_font = pygame.font.SysFont(roboto_path, 75)
        text = start_font.render('Press Enter to Start', True, (255, 255, 255))
        start_surface = pygame.Surface(self.surface_size)
        start_surface.fill((155, 101, 0))
        text_x, text_y = (self.surface_size[0] - text.get_size()[0]) // 2, (
                self.surface_size[1] - text.get_size()[1]) // 2
        start_surface.blit(text, (text_x, text_y))

        self.window.blit(start_surface, (0, 0))
        pygame.display.flip()

        if USING_WINDOWS:
            win = gw.getWindowsWithTitle('pygame window')[0]
            win.activate()

        start_screen = True
        while start_screen:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    start_screen = False

    def on_init(self):
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state,
                                                                         grid=self.env.env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": 1, "score": self.score})
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self._running = True

        if USING_WINDOWS:
            win = gw.getWindowsWithTitle('pygame window')[0]
            win.activate()

    def on_event(self, event):
        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == K_UP:
                action = Direction.NORTH
            elif pressed_key == K_RIGHT:
                action = Direction.EAST
            elif pressed_key == K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == K_LEFT:
                action = Direction.WEST
            elif pressed_key == K_SPACE:
                action = Action.INTERACT
            elif pressed_key == K_s:
                action = Action.STAY
            else:
                action = Action.STAY
            self.human_action = Action.ACTION_TO_INDEX[action]

        if event.type == pygame.QUIT:
            self._running = False

    def step_env(self, agent_action):
        prev_state = self.env.state

        obs, reward, done, info = self.env.step(agent_action)

        collision = self.env.mdp.prev_step_was_collision
        if collision:
            self.num_collisions += 1

        # Log data
        curr_reward = sum(info['sparse_r_by_agent'])
        self.score += curr_reward
        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(self.env.get_joint_action()),
            # TODO get teammate action from env to create joint_action json.dumps(joint_action.item()),
            "reward": curr_reward,
            "time_left": max((self.env.env.horizon - self.curr_tick) / self.fps, 0),
            "score": self.score,
            "time_elapsed": self.curr_tick / self.fps,
            "cur_gameloop": self.curr_tick,
            "layout": self.env.env.mdp.terrain_mtx,
            "layout_name": self.layout_name,
            "trial_id": self.trial_id,
            "user_id": self.user_id,
            "dimension": (self.x, self.y, self.surface_size, self.tile_size, self.grid_shape, self.hud_size),
            "Unix_timestamp": time.time(),
            "LSL_timestamp": local_clock(),
            "agent": self.teammate_name,
            "p_idx": self.p_idx,
            "collision": collision,
            "num_collisions": self.num_collisions
        }
        trans_str = json.dumps(transition)
        if self.outlet is not None:
            self.outlet.push_sample([trans_str])

        if self.collect_trajectory:
            self.trajectory.append(transition)
        return done

    def on_render(self, pidx=None):
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.env.state,
                                                                         grid=self.env.env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": self.curr_tick,
                                                                                   "score": self.score})
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        # Save screenshot
        # pygame.image.save(self.window, f"screenshots/screenshot_{self.curr_tick}.png")

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        self.start_screen()
        self.on_init()
        sleep_time = 1000 // (self.fps or 5)

        on_reset = True
        while (self._running):
            if self.agent == 'human':

                if self.human_action is None:
                    for event in pygame.event.get():
                        self.on_event(event)
                    pygame.event.pump()

                action = self.human_action if self.human_action is not None else Action.ACTION_TO_INDEX[Action.STAY]
            else:
                obs = self.env.get_obs(self.env.p_idx, on_reset=False)
                action = self.agent.predict(obs, state=self.env.state, deterministic=False)[0]
                get_conf = getattr(self.agent, "get_confidence", None)
                if callable(get_conf):
                    conf = get_conf(obs)
                    print(f"Current Confidence: {conf}")
                    if conf > 0.2:
                        pygame.time.wait(2)

            done = self.step_env(action)
            self.human_action = None
            if True or self.curr_tick < 200:
                pygame.time.wait(sleep_time)
            else:
                pygame.time.wait(1000)
            self.on_render()
            self.curr_tick += 1

            if done:
                self._running = False

        self.on_cleanup()
        print(f'Trial finished in {self.curr_tick} steps with total reward {self.score}')

    def save_trajectory(self, data_path):
        df = pd.DataFrame(self.trajectory)
        df.to_pickle(data_path / f'{self.layout_name}.{self.trial_id}.pickle')


class MultiOvercookedGUI:
    """Class to run n Overcooked Gridworld games. Adapted from OvercookedGUI for multiple rooms simultaneously.
    """

    def __init__(self, args, layout_name=None, agents=None, teammates=None, p_idx=0, horizon=400,
                 trial_id=None, user_id=None, stream=None, outlet=None, fps=5):
        self.n_rooms = len(agents)
        self.x = None
        self._running = True
        self._display_surf = None
        self.args = args
        self.layout_name = layout_name or 'asymmetric_advantages'

        self.agents = agents
        self.p_idx = p_idx

        self.teammate_names = ["None"] * self.n_rooms
        self.envs = []
        for i in range(self.n_rooms):
            env = OvercookedGymEnv(layout_name=self.layout_name, args=args, ret_completed_subtasks=True,
                                   is_eval_env=True, horizon=horizon)

            if teammates is not None:
                env.set_teammate(teammates[i])
                self.teammate_names[i] = teammates[i].name

            env.reset(p_idx=self.p_idx)
            if self.agents[i] != 'human':
                self.agents[i].set_idx(self.p_idx, env, is_hrl=isinstance(self.agents[i], HierarchicalRL), tune_subtasks=False)

            self.envs.append(env)

        self.grid_shape = self.envs[0].grid_shape
        self.trial_id = trial_id
        self.user_id = user_id
        self.fps = fps

        self.scores = [0] * self.n_rooms
        self.curr_tick = 1
        self.num_collisions = 0
        self.human_action = None
        self.data_path = args.base_dir / args.data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.tile_size = 50

        self.info_stream = stream
        self.outlet = outlet
        # Currently unused, but keeping in case we need it in the future.
        self.collect_trajectory = False

        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.envs[0].state,
                                                                         grid=self.envs[0].env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": 1, "score": 0})

        self.room_size = surface.get_size()

        self.proxy_idx = 0

    def start_screen(self):
        pygame.init()
        surface = StateVisualizer(tile_size=self.tile_size).render_state(self.envs[0].state,
                                                                         grid=self.envs[0].env.mdp.terrain_mtx,
                                                                         hud_data={"timestep": 1, "score": 0})

        self.room_size = surface.get_size()
        self.surface_size = (self.room_size[0] * self.n_rooms, self.room_size[1])
        self.x, self.y = (1920 - self.surface_size[0]) // 2, (1080 - self.surface_size[1]) // 2
        # self.grid_shape = self.env.mdp.shape
        self.hud_size = self.room_size[1] - (self.grid_shape[1] * self.tile_size)
        environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.x, self.y)

        self.window = pygame.display.set_mode(self.surface_size, HWSURFACE | DOUBLEBUF | RESIZABLE)

        pygame.font.init()
        start_font = pygame.font.SysFont(roboto_path, 75)
        text = start_font.render('Press Enter to Start', True, (255, 255, 255))
        start_surface = pygame.Surface(self.surface_size)
        start_surface.fill((155, 101, 0))
        text_x, text_y = (self.surface_size[0] - text.get_size()[0]) // 2, (
                self.surface_size[1] - text.get_size()[1]) // 2
        start_surface.blit(text, (text_x, text_y))

        self.window.blit(start_surface, (0, 0))
        pygame.display.flip()

        if USING_WINDOWS:
            win = gw.getWindowsWithTitle('pygame window')[0]
            win.activate()

        start_screen = True
        while start_screen:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    start_screen = False

    def on_init(self):
        self.on_render()

        self._running = True
        if USING_WINDOWS:
            windows = gw.getWindowsWithTitle('pygame window')
            if len(windows) > 0:
                win = windows[0]
                win.activate()

    def on_event(self, event):
        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']
            action = None

            if pressed_key == K_UP:
                action = Direction.NORTH
            elif pressed_key == K_RIGHT:
                action = Direction.EAST
            elif pressed_key == K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == K_LEFT:
                action = Direction.WEST
            elif pressed_key == K_SPACE:
                action = Action.INTERACT
            elif pressed_key == K_s:
                action = Action.STAY
            else:
                action = Action.STAY
            self.human_action = Action.ACTION_TO_INDEX[action]

        if event.type == pygame.QUIT:
            self._running = False

    def step_env(self, agent_action, room_idx, unstick=True):
        prev_state = self.envs[room_idx].state

        obs, reward, done, info = self.envs[room_idx].step(agent_action, unstick=unstick)

        collision = self.envs[room_idx].mdp.prev_step_was_collision
        if collision:
            self.num_collisions += 1

        # Log data
        curr_reward = sum(info['sparse_r_by_agent'])
        self.scores[room_idx] += curr_reward
        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(self.envs[room_idx].get_joint_action()),
            # TODO get teammate action from env to create joint_action json.dumps(joint_action.item()),
            "reward": curr_reward,
            "time_left": max((self.envs[room_idx].env.horizon - self.curr_tick) / self.fps, 0),
            "score": self.scores[room_idx],
            "time_elapsed": self.curr_tick / self.fps,
            "cur_gameloop": self.curr_tick,
            "layout": self.envs[room_idx].env.mdp.terrain_mtx,
            "layout_name": self.layout_name,
            "trial_id": self.trial_id,
            "user_id": self.user_id,
            "Unix_timestamp": time.time(),
            "LSL_timestamp": local_clock(),
            "agent": self.teammate_names[room_idx],
            "p_idx": self.p_idx,
            "collision": collision,
            "num_collisions": self.num_collisions
        }
        trans_str = json.dumps(transition)
        if self.outlet is not None:
            self.outlet.push_sample([trans_str])

        # if self.collect_trajectory:
        #     self.trajectory.append(transition)
        return done

    def on_render(self, flip=True):
        for i, env in enumerate(self.envs):
            surface = StateVisualizer(tile_size=self.tile_size).render_state(env.state,
                                                                             grid=env.env.mdp.terrain_mtx,
                                                                             hud_data={"agent": self.agents[i].name,
                                                                                       "timestep": self.curr_tick,
                                                                                       "score": self.scores[i]})
            # self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
            self.window.blit(surface, (self.room_size[0] * i, 0))

        if flip:
            pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        self.start_screen()
        self.on_init()
        sleep_time = 1000 // (self.fps or 5)

        while self._running:
            done = self.step(sleep_time)
            if done:
                self._running = False

        self.on_cleanup()
        print(f'Trial finished in {self.curr_tick} steps with total rewards {self.scores}')

    def step(self, sleep_time):
        done = False
        for i, env in enumerate(self.envs):  # Loop through rooms
            if self.agents[i] == 'human':
                if self.human_action is None:
                    for event in pygame.event.get():
                        self.on_event(event)
                    pygame.event.pump()

                action = self.human_action if self.human_action is not None else Action.ACTION_TO_INDEX[Action.STAY]
            else:
                action = self.get_action(i)

            done = done or self.step_env(action, i, unstick=i != 0)  # Unstick if we are not the human proxy agent

        self.human_action = None
        pygame.time.wait(sleep_time)
        self.on_render()
        self.curr_tick += 1

        return done

    def get_action(self, idx):
        obs = self.envs[idx].get_obs(self.envs[idx].p_idx, on_reset=False)
        return self.agents[idx].predict(obs, state=self.envs[idx].state, deterministic=False)[0]

    def save_trajectory(self, data_path):
        df = pd.DataFrame(self.trajectory)
        df.to_pickle(data_path / f'{self.layout_name}.{self.trial_id}.pickle')


class HandoverOvercookedGUI(MultiOvercookedGUI):
    def __init__(self, args, layout_name=None, ai_agents=None, teammates=None, p_idx=0, horizon=400, save_name=None,
                 trial_id=None, user_id=None, stream=None, outlet=None, fps=5, threshold=np.Inf, display=True):
        agents = [HumanProxy('proxy', args)] + ai_agents
        if teammates is not None:
            teammates = [None] + teammates
        super().__init__(args, layout_name=layout_name, agents=agents, teammates=teammates, p_idx=p_idx,
                         horizon=horizon, trial_id=trial_id, user_id=user_id, stream=stream, outlet=outlet, fps=fps)

        self.display = display

        self.proxy_idx = 0
        self.highlight_surface = pygame.Surface((self.room_size[0], 20))
        pygame.draw.rect(self.highlight_surface, (153, 255, 153), pygame.Rect(0, 0, self.room_size[0], 20))

        self.reward_record = np.zeros((horizon+1, self.n_rooms))
        self.threshold = threshold
        self.handover_steps = 0

        self.save_name = save_name

    def start_screen(self):
        if not self.display:
            return

        super().start_screen()

    def on_render(self, flip=True):  # Override on_render to highlight our human proxy agent's current room
        if not self.display:
            return

        super().on_render(flip=False)

        self.window.blit(self.highlight_surface, (self.room_size[0] * self.proxy_idx, self.room_size[1] - 20))
        pygame.display.flip()

        if self.save_name is not None:
            pygame.image.save(self.window, f"screenshots/{self.save_name}_{self.curr_tick}.png")

    def on_execute(self):
        super().on_execute()
        return self.reward_record, self.handover_steps / self.curr_tick

    def get_action(self, idx):
        if idx == 0 and self.proxy_idx != 0:  # If we're asking for a room 0 action and proxy agent is away, do nothing
            return Action.ACTION_TO_INDEX[Action.STAY]

        obs = self.envs[idx].get_obs(self.envs[idx].p_idx, on_reset=False)
        if idx == self.proxy_idx:  # If the proxy agent is currently in this room, do their action
            return self.agents[0].predict(obs, state=self.envs[idx].state, deterministic=False)[0]
        return self.agents[idx].predict(obs, state=self.envs[idx].state, deterministic=False)[0]

    def step(self, sleep_time):
        done = super().step(sleep_time if self.display else 0)
        self.evaluate_handover()
        self.reward_record[self.curr_tick-1, :] = self.scores

        return done

    def evaluate_handover(self):
        confidences = []
        for i in range(1, self.n_rooms):
            obs = self.envs[i].get_obs(self.envs[i].p_idx, on_reset=False)
            confidences.append(self.agents[i].get_confidence(obs))

        if confidences[0] > self.threshold:
            self.proxy_idx = 1
            self.handover_steps += 1
        else:
            self.proxy_idx = 0

