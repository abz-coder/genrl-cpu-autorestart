from unittest import TestCase
from genrl.rewards import DefaultRewardManager, RewardFnStore, RoundRewardFnStore


class TestRewardManager(TestCase):
    def setUp(self) -> None:
        self.rm = DefaultRewardManager(RewardFnStore(1, []))

    def test_setters(self):
        self.rm.round = 3
        self.assertEqual(self.rm.round, 3)
        self.rm.stage = 2
        self.assertEqual(self.rm.stage, 2)
        self.rm.rewards = [1, 2, 3]
        self.assertEqual(self.rm.rewards, [1, 2, 3])

    def test_getters(self):
        self.rm.set_round_stage(3, 2)
        self.assertEqual(self.rm.round, 3)
        self.assertEqual(self.rm.stage, 2)
        self.assertEqual(self.rm.rewards, [])

    def test_set_round_stage(self):
        self.rm.set_round_stage(11, 12)
        self.assertEqual(self.rm.round, 11)
        self.assertEqual(self.rm.stage, 12)

    def test_reset(self):
        self.rm.set_round_stage(3, 2)
        self.rm.reset()
        self.assertEqual(self.rm.round, 3)
        self.assertEqual(self.rm.stage, 0)
        self.assertEqual(self.rm.rewards, [])
        
    def test_dispatch_reward_fn(self):
        round_store = RoundRewardFnStore(3, [lambda x: x + 1, lambda x: x + 2, lambda x: x + 3])
        reward_store = RewardFnStore(1, [round_store])
        rm = DefaultRewardManager(reward_store)
        reward_fn = rm.dispatch_reward_fn(0, 2)
        reward = reward_fn(10)
        self.assertEqual(reward, 13)

    def test_call(self):
        round_store = RoundRewardFnStore(3, [lambda x: x + 1, lambda x: x + 2, lambda x: x + 3])
        reward_store = RewardFnStore(1, [round_store])
        rm = DefaultRewardManager(reward_store)
        reward = rm(0, 1, 10)
        self.assertEqual(reward, 12)
        self.assertEqual(rm.rewards, [12])
        