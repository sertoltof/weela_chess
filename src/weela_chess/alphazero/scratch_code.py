# # todo move to a ttt specific class
# def rando_dummy_play(self):
#     memory: list[tuple[MCTSStateMachine, NDArray]] = []
#     state_machine = self.root_state
#     azero_player = np.random.choice([-1, 1])
#
#     while True:
#         if state_machine.whose_turn == azero_player:
#             action_probs = self.mcts.search(state_machine)
#             memory.append((state_machine, action_probs))
#
#             temperature_action_probs = action_probs ** (1 / self.train_params.temperature)
#             temperature_action_probs = temperature_action_probs / np.sum(temperature_action_probs)
#             action = np.random.choice(self.root_state.action_size, p=temperature_action_probs)  # change to p=temperature_action_probs
#         else:
#             action = np.random.choice(state_machine.valid_actions())
#
#         state_machine = state_machine.take_action(action)
#         value = state_machine.state_value()
#         is_terminal = state_machine.check_is_over()
#
#         if is_terminal:
#             return_memory = []
#             for hist_state, hist_action_probs in memory:
#                 return_memory.append((
#                     hist_state.get_encoded_state(),
#                     hist_action_probs,
#                     value
#                 ))
#             return return_memory