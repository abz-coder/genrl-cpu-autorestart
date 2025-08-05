from unittest import TestCase

from genrl.state import GameNode, DefaultGameTree, WorldState

class TestGameTree(TestCase):
    def setUp(self) -> None:
        self.node = GameNode(0, 0)
        self.gt = DefaultGameTree(root_states=WorldState(environment_states='puppies', opponent_states='und', personal_states='kittens'))

    def test_node(self) -> None:
        #Check that default child and parents are empty lists 
        self.assertTrue(self.node["parents"] == [])
        self.assertTrue(self.node["children"] == [])
        #Append children
        self.node.append_child(GameNode(1, 0))
        self.node.append_child(GameNode(1, 1))
        self.node.append_child(GameNode(0, 2))
        self.assertEqual((self.node["children"][0]["stage"],self.node["children"][0]["node_idx"]), (1,0))
        self.assertEqual((self.node["children"][1]["stage"],self.node["children"][1]["node_idx"]), (1,1))
        self.assertEqual((self.node["children"][2]["stage"],self.node["children"][2]["node_idx"]), (0,2))
        self.assertTrue(len(self.node["children"]) == 3)
        #Append parents
        self.node.append_parent(GameNode(0, 1))
        self.node.append_parent(GameNode(-1, 2))
        self.node.append_parent(GameNode(-10, 10))
        self.assertEqual((self.node["parents"][0]["stage"],self.node["parents"][0]["node_idx"]), (0,1))
        self.assertEqual((self.node["parents"][1]["stage"],self.node["parents"][1]["node_idx"]), (-1,2))
        self.assertEqual((self.node["parents"][2]["stage"],self.node["parents"][2]["node_idx"]), (-10,10))
        self.assertTrue(len(self.node["parents"]) == 3)
        #Set as special nodes
        self.node = GameNode(0, 0)
        self.node.set_as_root()
        self.assertTrue(self.node["parents"] == None)
        self.assertTrue(self.node._is_root_node())
        self.node.set_as_terminal()
        self.assertTrue(self.node["children"] == None)
        self.assertTrue(self.node._is_leaf_node())

    def test_append_node_states(self) -> None:
        example_states = WorldState(environment_states='Example env state new', opponent_states='Example op state new', personal_states=['e','x','a','m','p','l','e',3])
        #No pruner
        pruning_functions = {"environment_pruner": None, "opponent_pruner": None, "personal_pruner": None}
        self.gt.append_node_states(stage=0, node_idx=0, states=example_states, pruning_functions=pruning_functions)
        self.assertEqual(self.gt[0][0].world_state.environment_states, example_states.environment_states)
        self.assertEqual(self.gt[0][0].world_state.opponent_states, example_states.opponent_states)
        self.assertEqual(self.gt[0][0].world_state.personal_states, example_states.personal_states)
        #With pruner
        def toy_pruner(x):
            return x[0]
        pruning_functions = {"environment_pruner": toy_pruner, "opponent_pruner": toy_pruner, "personal_pruner": toy_pruner}
        self.gt.append_node_states(stage=0, node_idx=0, states=example_states, pruning_functions=pruning_functions)
        self.assertEqual(self.gt[0][0].world_state.environment_states, toy_pruner(example_states.environment_states))
        self.assertEqual(self.gt[0][0].world_state.opponent_states, toy_pruner(example_states.opponent_states))
        self.assertEqual(self.gt[0][0].world_state.personal_states, toy_pruner(example_states.personal_states))

    def test_append_node_actions(self) -> None:
        example_actions = ['Action 1', 'Action 2', 3]
        self.gt.append_node_actions(stage=0, node_idx=0, actions=example_actions)
        self.assertEqual(self.gt[0][0]["actions"], example_actions)
        self.assertTrue(len(self.gt[0][0]["actions"])==len(example_actions))

    def test_commit_actions_from_stage(self) -> None:
        example_actions = ['Action 1', 'Action 2', 3]
        self.gt.append_node_actions(stage=0, node_idx=0, actions=example_actions)
        #No branching fxns
        example_branching_fxns = {"terminal_node_decision_function": None, "stage_inheritance_function": None}
        self.gt.commit_actions_from_stage(stage=0,tree_branching_functions=example_branching_fxns)
        self.assertEqual(len(self.gt[0][0]["children"]), len(self.gt[0][0]["actions"]))
        self.assertTrue(self.gt.metadata['max_depth'] == 1)
        self.assertTrue(self.gt.metadata['num_nodes'] == 4)
        for child in self.gt[0][0]["children"]:
            self.assertEqual(child["stage"], self.gt.metadata['max_depth'])
            self.assertEqual(child.world_state.environment_states, self.gt[0][0].world_state.environment_states)
            self.assertEqual(child.world_state.opponent_states, self.gt[0][0].world_state.opponent_states)
            self.assertTrue(child.world_state.personal_states in self.gt[0][0].world_state.personal_states)
        #With stage_inheritance_function
        def toy_inheritance(x):
            stage_nodes = []
            for node in x:
                if node._is_leaf_node():
                    stage_nodes.append([])
                else:
                    stage_nodes.append([GameNode(stage=0, 
                                                node_idx=0, 
                                                world_state=WorldState(
                                                    environment_states='tuna', 
                                                    opponent_states='fish', 
                                                    personal_states='cans'), 
                                                actions=['have', 1, 'fish'])])
            return stage_nodes
        example_branching_fxns = {"terminal_node_decision_function": None, "stage_inheritance_function": toy_inheritance}
        self.gt.commit_actions_from_stage(stage=1,tree_branching_functions=example_branching_fxns)
        self.assertEqual(len(self.gt[1][0]["children"]), len(self.gt[1][1]["children"]))
        self.assertTrue(self.gt.metadata['max_depth'] == 2)
        self.assertTrue(self.gt.metadata['num_nodes'] == 7)
        for child in self.gt[1][0]["children"]:
            self.assertEqual(child["stage"], self.gt.metadata['max_depth'])
            self.assertEqual(child.world_state.environment_states, 'tuna')
            self.assertEqual(child.world_state.opponent_states, 'fish')
            self.assertEqual(child.world_state.personal_states, 'cans')
        #With terminal_node_decision_function
        example_branching_fxns = {"terminal_node_decision_function": None, "stage_inheritance_function": None}
        self.gt.commit_actions_from_stage(stage=2,tree_branching_functions=example_branching_fxns)
        self.assertTrue(self.gt.metadata['max_depth'] == 3)
        def toy_terminal_decider(x):
            terminal_nodes = []
            for node in x:
                if isinstance(node.actions, int):
                    terminal_nodes.append(node)
            return terminal_nodes
        example_branching_fxns = {"terminal_node_decision_function": toy_terminal_decider, "stage_inheritance_function": toy_inheritance}
        self.gt.commit_actions_from_stage(stage=3,tree_branching_functions=example_branching_fxns)
        self.assertTrue(self.gt[3][1]._is_leaf_node())
        self.assertTrue(not self.gt[3][0]._is_leaf_node())
        self.assertTrue(self.gt.metadata['max_depth'] == 4)
        #Checking all_branches_terminal
        self.assertTrue(not self.gt.all_branches_terminal())
        def toy_terminal_decider(x):
            terminal_nodes = []
            for node in x:
                terminal_nodes.append(node)
            return terminal_nodes
        example_branching_fxns = {"terminal_node_decision_function": toy_terminal_decider, "stage_inheritance_function": toy_inheritance}
        self.gt.commit_actions_from_stage(stage=4,tree_branching_functions=example_branching_fxns)
        self.assertTrue(self.gt.all_branches_terminal())

        
        
        
        

