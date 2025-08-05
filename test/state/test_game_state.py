from unittest import TestCase
import torch

from genrl.state import GameState, WorldState


class TestGameState(TestCase):
    def setUp(self) -> None:
        self.batch_size = 5
        self.state = GameState(0, 0)
        self.swarm_size = 1

    def test_init_game(self) -> None:
        round_data_raw = torch.randint(0, 100, (self.batch_size, 3))
        round_data = [(i, WorldState(round_data_raw[i][0], round_data_raw[i][1], round_data_raw[i][2])) for i in range(len(round_data_raw.tolist()))]
        self.state._init_game(round_data)
        self.assertEqual(self.state.world_state_pruners, {"environment_pruner": None, "opponent_pruner": None, "personal_pruner": None})
        self.assertEqual(self.state.tree_branching_functions, {"terminal_node_decision_function": None, "stage_inheritance_function": None})
        self.assertEqual(self.state.batch_size, self.batch_size)
        self.assertEqual(len(self.state.trees), self.swarm_size)
        for agent in self.state.trees:
            self.assertEqual(len(self.state.trees[agent]), self.batch_size)
            for batch in range(self.batch_size):
                self.assertEqual(self.state.trees[agent][batch].metadata['root_node'].world_state.environment_states, round_data[batch][1].environment_states)
                self.assertEqual(self.state.trees[agent][batch].metadata['root_node'].world_state.opponent_states, round_data[batch][1].opponent_states)
                self.assertEqual(self.state.trees[agent][batch].metadata['root_node'].world_state.personal_states, round_data[batch][1].personal_states)
    
    def test_advance_round(self) -> None:
        round_data_raw = torch.randint(0, 100, (self.batch_size, 3))
        round_data = [(i, round_data_raw[i][0],round_data_raw[i][1],round_data_raw[i][2]) for i in range(len(round_data_raw.tolist()))]
        self.state.advance_round(round_data)
        self.assertEqual(self.state.round, 1)
        self.assertEqual(self.state.stage, 0)

    def test_get_stage_state(self) -> None:
        round_data_raw = torch.randint(0, 100, (self.batch_size, 3))
        round_data = [(i, WorldState(round_data_raw[i][0], round_data_raw[i][1], round_data_raw[i][2])) for i in range(len(round_data_raw.tolist()))]
        self.state.advance_round(round_data)
        states = self.state.get_stage_state(stage_num=0) #[Agents][Batch][Node Idx in Stage][World State]
        self.assertEqual(len(states), self.swarm_size)
        for agent in self.state.trees:
            self.assertEqual(len(states[agent]), self.batch_size)
            for batch in range(self.batch_size):
                self.assertEqual(len(states[agent][batch]), 1)
                self.assertEqual(states[agent][batch][0].environment_states, round_data[batch][1].environment_states)
                self.assertEqual(states[agent][batch][0].opponent_states, round_data[batch][1].opponent_states)
                self.assertEqual(states[agent][batch][0].personal_states, round_data[batch][1].personal_states)
                
    def test_get_stage_actions(self) -> None:
        round_data_raw = torch.randint(0, 100, (self.batch_size, 3))
        round_data = [(i, round_data_raw[i][0],round_data_raw[i][1],round_data_raw[i][2]) for i in range(len(round_data_raw.tolist()))]
        self.state.advance_round(round_data)
        states = self.state.get_stage_actions(stage_num=0) #[Agents][Batch][Node Idx in Stage]
        self.assertEqual(len(states), self.swarm_size)
        for agent in self.state.trees:
            self.assertEqual(len(states[agent]), self.batch_size)
            for batch in range(self.batch_size):
                self.assertEqual(len(states[agent][batch]), 1)
                self.assertEqual(states[agent][batch][0], None)

    def test_get_latest_state(self) -> None:
        round_data_raw = torch.randint(0, 100, (self.batch_size, 3))
        round_data = [(i, round_data_raw[i][0],round_data_raw[i][1],round_data_raw[i][2]) for i in range(len(round_data_raw.tolist()))]
        self.state.advance_round(round_data)
        states = self.state.get_latest_state() #[Agents][Batch][Node Idx in Stage][World State]
        self.assertEqual(len(states), self.swarm_size)
        for agent in self.state.trees:
            self.assertEqual(len(states[agent]), self.batch_size)
            for batch in range(self.batch_size):
                self.assertEqual(len(states[agent][batch]), 1)
                self.assertEqual(states[agent][batch][0], round_data[batch][1])
                self.assertEqual(states[agent][batch][0], round_data[batch][1])
                self.assertEqual(states[agent][batch][0], round_data[batch][1])
    
    def test_get_latest_actions(self) -> None:
        round_data_raw = torch.randint(0, 100, (self.batch_size, 3))
        round_data = [(i, round_data_raw[i][0],round_data_raw[i][1],round_data_raw[i][2]) for i in range(len(round_data_raw.tolist()))]
        self.state.advance_round(round_data)
        states = self.state.get_latest_actions() #[Agents][Batch][Node Idx in Stage]
        self.assertEqual(len(states), self.swarm_size)
        for agent in self.state.trees:
            self.assertEqual(len(states[agent]), self.batch_size)
            for batch in range(self.batch_size):
                self.assertEqual(len(states[agent][batch]), 1)
                self.assertEqual(states[agent][batch][0], None)

    def test_append_actions(self) -> None:
        round_data_raw = torch.randint(0, 100, (self.batch_size, 3))
        round_data = [(i, round_data_raw[i][0],round_data_raw[i][1],round_data_raw[i][2]) for i in range(len(round_data_raw.tolist()))]
        self.state.advance_round(round_data)
        actions = ['generation', 'being', 'appended']
        agent_actions = {agent:[[actions for node in self.state.trees[agent][batch][0]] for batch in range(self.batch_size)] for agent in self.state.trees}
        self.state.append_actions(agent_actions=agent_actions)
        states = self.state.get_stage_actions(stage_num=0)
        for agent in self.state.trees:
            for batch in range(self.batch_size):
                self.assertEqual(states[agent][batch][0], actions)
                self.assertEqual(self.state.trees[agent][batch][0][0]['actions'], actions)

    def test_advance_stage(self) -> None:
        round_data_raw = torch.randint(0, 100, (self.batch_size, 3))
        round_data = [(i, round_data_raw[i][0],round_data_raw[i][1],round_data_raw[i][2]) for i in range(len(round_data_raw.tolist()))]
        self.state.advance_round(round_data)
        new_stage_data = torch.randint(0, 100, (self.batch_size, 3))
        world_states = {agent:[[(new_stage_data[batch][0],new_stage_data[batch][1],new_stage_data[batch][2]) for node in self.state.trees[agent][batch][1]] for batch in range(self.batch_size)] for agent in self.state.trees} 
        self.state.advance_stage(world_states)
        self.assertEqual(self.state.stage, 1)
        for agent in self.state.trees:
            for batch in range(self.batch_size):
                for node in range(len(self.state.trees[agent][batch][self.state.stage])):
                    self.assertEqual(self.state.trees[agent][batch][node]['environment_states'], world_states[agent][batch][node][0])
                    self.assertEqual(self.state.trees[agent][batch][node]['opponent_states'], world_states[agent][batch][node][1])
                    self.assertEqual(self.state.trees[agent][batch][node]['personal_states'], world_states[agent][batch][node][2])
