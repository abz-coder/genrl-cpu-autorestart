from unittest import TestCase
from genrl.rewards import RoundRewardFnStore, RewardFnStore


class TestRoundRewardStore(TestCase):
    def setUp(self) -> None:
        self.store_1_stage = RoundRewardFnStore(1, [])
        self.store_2_stage = RoundRewardFnStore(2, [])
        self.store_3_stage = RoundRewardFnStore(3, [lambda x: x + 1, lambda x: x + 2, lambda x: x + 3])

    def test_init(self):
        with self.assertRaises(ValueError):
            RoundRewardFnStore(0, [])

        self.assertEqual(self.store_1_stage.num_stages, 1)
        results = [f(10) for f in self.store_1_stage.reward_fns]
        self.assertEqual(results, [0])

        self.assertEqual(self.store_2_stage.num_stages, 2)
        results = [f(10) for f in self.store_2_stage.reward_fns]
        self.assertEqual(results, [0, 0])

        self.assertEqual(self.store_3_stage.num_stages, 3)
        results = [f(10) for f in self.store_3_stage.reward_fns]
        self.assertEqual(results, [11, 12, 13])


class TestRewardStore(TestCase):
    def setUp(self) -> None:
        self.store_1_round = RewardFnStore(1, [])
        self.store_2_round = RewardFnStore(2, [])

        round_store = RoundRewardFnStore(3, [lambda x: x + 1, lambda x: x + 2, lambda x: x + 3])
        self.store_3_round_partial = RewardFnStore(3, [round_store])
        self.store_3_round_complete = RewardFnStore(3, [round_store] * 3)

    def test_init(self):
        with self.assertRaises(ValueError):
            RewardFnStore(0, [])

        self.assertEqual(self.store_1_round.max_rounds, 1)
        results = [f(10) for f in self.store_1_round.reward_fn_stores[0].reward_fns]
        self.assertEqual(results, [0])

        self.assertEqual(self.store_2_round.max_rounds, 2)
        results = [f(10) for store in self.store_2_round.reward_fn_stores for f in store.reward_fns]
        self.assertEqual(results, [0, 0])
        
        self.assertEqual(self.store_3_round_partial.max_rounds, 3)
        results = [f(10) for store in self.store_3_round_partial.reward_fn_stores for f in store.reward_fns]
        self.assertEqual(results, [11, 12, 13] * 3)

        self.assertEqual(self.store_3_round_complete.max_rounds, 3)
        results = [f(10) for store in self.store_3_round_complete.reward_fn_stores for f in store.reward_fns]
        self.assertEqual(results, [11, 12, 13] * 3)
