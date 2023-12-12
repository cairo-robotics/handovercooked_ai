from typing import Tuple, Union

import torch as th
from stable_baselines3.common import distributions as sb3_distributions

from agents import OAIAgent
from overcooked_ai_py.agents.agent import GreedyHumanModel
from collections import defaultdict


class HumanProxy(OAIAgent):
    def __init__(self, name, args, hl_boltzmann=False, ll_boltzmann=False, hl_temp=1, ll_temp=1):
        super().__init__(name, args)
        self.hl_boltzmann = hl_boltzmann
        self.ll_boltzmann = ll_boltzmann
        self.hl_temp = hl_temp
        self.ll_temp = ll_temp

        self.proxy = None

    def set_idx(self, p_idx, env, is_hrl=False, output_message=True, tune_subtasks=False):
        self.p_idx = p_idx
        self.proxy = GreedyHumanModelSingle(env.env.mlam, self.hl_boltzmann, self.ll_boltzmann,
                                            self.hl_temp, self.ll_temp)
        self.proxy.set_mdp(env.mdp)
        self.proxy.reset()
        self.proxy.set_agent_index(p_idx)

    def predict(self, obs: th.Tensor, state=None, episode_start=None, deterministic: bool = False):
        action = self.proxy.action(state)
        print(action[1]['action_probs'])
        return action[1]['action_probs'].argmax(), None

    def get_distribution(self, obs: th.Tensor) -> Union[th.distributions.Distribution, sb3_distributions.Distribution]:
        pass

    def update_env(self, env):
        self.proxy.update_env(env.env)


class GreedyHumanModelSingle(GreedyHumanModel):

    def ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
            [do X] at location [Y]
        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of greedy heuristics based on the current state.
        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        player = state.players[self.agent_index]
        # other_player = state.players[1 - self.agent_index]
        am = self.mlam

        counter_objects = self.mlam.mdp.get_counter_objects_dict(
            state, list(self.mlam.mdp.terrain_pos_dict["X"])
        )
        pot_states_dict = self.mlam.mdp.get_pot_states(state)

        if not player.has_object():
            ready_soups = pot_states_dict["ready"]
            cooking_soups = pot_states_dict["cooking"]

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = False # (other_player.has_object()
                    # and other_player.get_object().name == "dish")


            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                assert len(state.all_orders) == 1 and list(
                    state.all_orders[0].ingredients
                ) == ["onion", "onion", "onion"], (
                        "The current mid level action manager only support 3-onion-soup order, but got orders"
                        + str(state.all_orders)
                )
                next_order = list(state.all_orders)[0]
                soups_ready_to_cook_key = "{}_items".format(
                    len(next_order.ingredients)
                )
                soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
                if soups_ready_to_cook:
                    only_pot_states_ready_to_cook = defaultdict(list)
                    only_pot_states_ready_to_cook[
                        soups_ready_to_cook_key
                    ] = soups_ready_to_cook
                    # we want to cook only soups that has same len as order
                    motion_goals = am.start_cooking_actions(
                        only_pot_states_ready_to_cook
                    )
                else:
                    motion_goals = am.pickup_onion_actions(counter_objects)
                # it does not make sense to have tomato logic when the only possible order is 3 onion soup (see assertion above)
                # elif 'onion' in next_order:
                #     motion_goals = am.pickup_onion_actions(counter_objects)
                # elif 'tomato' in next_order:
                #     motion_goals = am.pickup_tomato_actions(counter_objects)
                # else:
                #     motion_goals = am.pickup_onion_actions(counter_objects) + am.pickup_tomato_actions(counter_objects)

        else:
            player_obj = player.get_object()

            if player_obj.name == "onion":
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == "tomato":
                motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)

            elif player_obj.name == "dish":
                motion_goals = am.pickup_soup_with_dish_actions(
                    pot_states_dict, only_nearly_ready=True
                )

            elif player_obj.name == "soup":
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]

        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [
                mg
                for mg in motion_goals
                if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                    player.pos_and_or, mg
                )
            ]
            assert len(motion_goals) != 0

        return motion_goals

    def update_env(self, new_env):
        self.mlam = new_env.mlam
        self.mdp = self.mlam.mdp