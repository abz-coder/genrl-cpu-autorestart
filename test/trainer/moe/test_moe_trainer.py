import os

from transformers import Qwen2Config, Qwen2ForCausalLM

from genrl.rewards import RewardManager
from genrl.state.game_state import GameState
from genrl.trainer.moe.moe_trainer import MixtureOfExpertsTrainer
import pytest

# Skip this entire test file
pytest.skip("Skipping all tests in test_distributed_moe_layer_utils.py", allow_module_level=True)

class DummyRewardManager(RewardManager):
    _REWARDS = None

    def update_rewards(self, game_state: GameState) -> None:
        # [agent][batch][stage]
        self._REWARDS = [
            [[0.0] + [1.0] * (len(sample) - 1) for sample in batch]
            for batch in game_state.outputs
        ]

    def reset(self) -> None:
        self._REWARDS = None

    def __call__(self):
        assert self._REWARDS is not None
        return self._REWARDS


def setup(path):
    model_path = os.path.join(path, "model")
    os.makedirs(model_path, exist_ok=True)
    config = Qwen2Config(
        hidden_size=64,
        vocab_size=50257,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=4,
        max_position_embeddings=128,
        intermediate_size=128,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(model_path)
    return model_path, "openai-community/gpt2"


def test_trainer_generate(tmp_path):
    model_path, tokenizer_path = setup(tmp_path)
    trainer = MixtureOfExpertsTrainer(
        model_path_or_name=model_path,
        tokenizer_path_or_name=tokenizer_path,
        epsilon_low=0.1,
        epsilon_high=0.2,
        max_new_tokens=16,
        num_return_sequences=2,
        num_experts_per_tok=1,
    )

    # [Agent][batch][turns][text]
    inputs = [
        [
            ["this is a cat and this is a dog", "this is a turn"],
            ["there is shed in the yard"],
        ],
        [["sample text"], ["more sample text", "text", "this is the last turn"]],
    ]
    outputs = trainer.generate(inputs)

    assert len(outputs) == 4
    assert all([len(output) == 2 for output in outputs])


def test_trainer_train(tmp_path):
    model_path, tokenizer_path = setup(tmp_path)
    trainer = MixtureOfExpertsTrainer(
        model_path_or_name=model_path,
        tokenizer_path_or_name=tokenizer_path,
        epsilon_low=0.1,
        epsilon_high=0.2,
        max_new_tokens=16,
        num_return_sequences=2,
        num_experts_per_tok=1,
    )
    outputs = [
        [
            ["this is a cat and this is a dog", "this is a turn"],
            ["there is shed in the yard"],
        ],
        [["sample text"], ["more sample text", "text", "this is the last turn"]],
    ]
    game_state = GameState(0, 0, None, outputs)
    reward_manager = DummyRewardManager()
    reward_manager.update_rewards(game_state)
    trainer.train(game_state, reward_manager)
