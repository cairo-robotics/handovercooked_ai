from agents import OAIAgent


class GreedyHumanModel(OAIAgent):
    """
    Agent that at each step selects a medium level action corresponding
    to the most intuitively high-priority thing to do

    NOTE: MIGHT NOT WORK IN ALL ENVIRONMENTS, for example forced_coordination.layout,
    in which an individual agent cannot complete the task on their own.
    Will work only in environments where the only order is 3 onion soup.
    """

    def __init__(self, mlam, hl_boltzmann_rational=False, ll_boltzmann_rational=False, hl_temp=1, ll_temp=1,
                 auto_unstuck=True):
        self.mlam = mlam
        self.mdp = self.mlam.mdp

        # Bool for perfect rationality vs Boltzmann rationality for high level and low level action selection
        self.hl_boltzmann_rational = hl_boltzmann_rational  # For choices among high level goals of same type
        self.ll_boltzmann_rational = ll_boltzmann_rational  # For choices about low level motion

        # Coefficient for Boltzmann rationality for high level action selection
        self.hl_temperature = hl_temp
        self.ll_temperature = ll_temp

        # Whether to automatically take an action to get the agent unstuck if it's in the same
        # state as the previous turn. If false, the agent is history-less, while if true it has history.
        self.auto_unstuck = auto_unstuck
        self.reset()

    def reset(self):
        super().reset()
        self.prev_state = None

    def actions(self, states, agent_indices):
        actions_and_infos_n = []
        for state, agent_idx in zip(states, agent_indices):
            self.set_agent_index(agent_idx)
            self.reset()
            actions_and_infos_n.append(self.action(state))
        return actions_and_infos_n

    def action(self, state):
        possible_motion_goals = self.ml_action(state)

        # Once we have identified the motion goals for the medium
        # level action we want to perform, select the one with lowest cost
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        chosen_goal, chosen_action, action_probs = self.choose_motion_goal(start_pos_and_or, possible_motion_goals)

        if self.ll_boltzmann_rational and chosen_goal[0] == start_pos_and_or[0]:
            chosen_action, action_probs = self.boltzmann_rational_ll_action(start_pos_and_or, chosen_goal)

        if self.auto_unstuck:
            # HACK: if two agents get stuck, select an action at random that would
            # change the player positions if the other player were not to move
            if self.prev_state is not None and state.players_pos_and_or == self.prev_state.players_pos_and_or:
                if self.agent_index == 0:
                    joint_actions = list(itertools.product(Action.ALL_ACTIONS, [Action.STAY]))
                elif self.agent_index == 1:
                    joint_actions = list(itertools.product([Action.STAY], Action.ALL_ACTIONS))
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    new_state, _ = self.mlam.mdp.get_state_transition(state, j_a)
                    if new_state.player_positions != self.prev_state.player_positions:
                        unblocking_joint_actions.append(j_a)
                # Getting stuck became a possiblity simply because the nature of a layout (having a dip in the middle)
                if len(unblocking_joint_actions) == 0:
                    unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[np.random.choice(len(unblocking_joint_actions))][
                    self.agent_index]
                action_probs = self.a_probs_from_action(chosen_action)

            # NOTE: Assumes that calls to the action method are sequential
            self.prev_state = state
        return chosen_action, {"action_probs": action_probs}

    def choose_motion_goal(self, start_pos_and_or, motion_goals):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """
        if self.hl_boltzmann_rational:
            possible_plans = [self.mlam.motion_planner.get_plan(start_pos_and_or, goal) for goal in motion_goals]
            plan_costs = [plan[2] for plan in possible_plans]
            goal_idx, action_probs = self.get_boltzmann_rational_action_idx(plan_costs, self.hl_temperature)
            chosen_goal = motion_goals[goal_idx]
            chosen_goal_action = possible_plans[goal_idx][0][0]
        else:
            chosen_goal, chosen_goal_action = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
            action_probs = self.a_probs_from_action(chosen_goal_action)
        return chosen_goal, chosen_goal_action, action_probs

    def get_boltzmann_rational_action_idx(self, costs, temperature):
        """Chooses index based on softmax probabilities obtained from cost array"""
        costs = np.array(costs)
        softmax_probs = np.exp(-costs * temperature) / np.sum(np.exp(-costs * temperature))
        action_idx = np.random.choice(len(costs), p=softmax_probs)
        return action_idx, softmax_probs

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def boltzmann_rational_ll_action(self, start_pos_and_or, goal, inverted_costs=False):
        """
        Computes the plan cost to reach the goal after taking each possible low level action.
        Selects a low level action boltzmann rationally based on the one-step-ahead plan costs.

        If `inverted_costs` is True, it will make a boltzmann "irrational" choice, exponentially
        favouring high cost plans rather than low cost ones.
        """
        future_costs = []
        for action in Action.ALL_ACTIONS:
            pos, orient = start_pos_and_or
            new_pos_and_or = self.mdp._move_if_direction(pos, orient, action)
            _, _, plan_cost = self.mlam.motion_planner.get_plan(new_pos_and_or, goal)
            sign = (-1) ** int(inverted_costs)
            future_costs.append(sign * plan_cost)

        action_idx, action_probs = self.get_boltzmann_rational_action_idx(future_costs, self.ll_temperature)
        return Action.ALL_ACTIONS[action_idx], action_probs

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
        other_player = state.players[1 - self.agent_index]
        am = self.mlam

        counter_objects = self.mlam.mdp.get_counter_objects_dict(state, list(self.mlam.mdp.terrain_pos_dict['X']))
        pot_states_dict = self.mlam.mdp.get_pot_states(state)


        if not player.has_object():
            ready_soups = pot_states_dict['ready']
            cooking_soups = pot_states_dict['cooking']

            soup_nearly_ready = len(ready_soups) > 0 or len(cooking_soups) > 0
            other_has_dish = other_player.has_object() and other_player.get_object().name == 'dish'

            if soup_nearly_ready and not other_has_dish:
                motion_goals = am.pickup_dish_actions(counter_objects)
            else:
                assert len(state.all_orders) == 1 and list(state.all_orders[0].ingredients) == ["onion", "onion", "onion"], \
                    "The current mid level action manager only support 3-onion-soup order, but got orders" \
                    + str(state.all_orders)
                next_order = list(state.all_orders)[0]
                soups_ready_to_cook_key = '{}_items'.format(len(next_order.ingredients))
                soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
                if soups_ready_to_cook:
                    only_pot_states_ready_to_cook = defaultdict(list)
                    only_pot_states_ready_to_cook[soups_ready_to_cook_key] = soups_ready_to_cook
                    # we want to cook only soups that has same len as order
                    motion_goals = am.start_cooking_actions(only_pot_states_ready_to_cook)
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

            if player_obj.name == 'onion':
                motion_goals = am.put_onion_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'tomato':
                motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)

            elif player_obj.name == 'dish':
                motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)

            elif player_obj.name == 'soup':
                motion_goals = am.deliver_soup_actions()

            else:
                raise ValueError()

        motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]

        if len(motion_goals) == 0:
            motion_goals = am.go_to_closest_feature_actions(player)
            motion_goals = [mg for mg in motion_goals if self.mlam.motion_planner.is_valid_motion_start_goal_pair(player.pos_and_or, mg)]
            assert len(motion_goals) != 0

        return motion_goals