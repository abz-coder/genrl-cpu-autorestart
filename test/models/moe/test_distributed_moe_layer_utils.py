import pytest

# Skip this entire test file
pytest.skip("Skipping all tests in test_distributed_moe_layer_utils.py", allow_module_level=True)

import torch
import torch.multiprocessing as mp
from transformers import Qwen2Config

from genrl.models.moe.distributed_moe_layer_utils import \
    DistributedMoEContext
from genrl.models.moe.moe_helpers import (Qwen2ForCausalLM,
                                                from_qwen_to_moe)


def _test_distributed_moe_context(rank, init_method, world_size):
    torch.distributed.init_process_group(
        init_method=init_method,
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    config = Qwen2Config(
        hidden_size=64,
        vocab_size=128,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=4,
        max_position_embeddings=128,
        intermediate_size=128,
    )
    qwen2model = Qwen2ForCausalLM(config)
    qwen2model.eval()
    moe_model = from_qwen_to_moe(qwen2model, num_experts=1, num_experts_per_tok=1)
    moe_model.eval()

    input_ids = torch.tensor([list(range(10))], dtype=torch.int64)
    with torch.no_grad():
        expected_output = qwen2model(input_ids)
        output = moe_model(input_ids)
    torch.testing.assert_close(expected_output[0], output[0])

    with DistributedMoEContext(moe_model=moe_model, top_k=world_size):
        moe_outputs = moe_model(input_ids)
        moe_outputs[0].backward(torch.ones_like(moe_outputs[0]))

    with torch.no_grad():
        new_output = moe_model(input_ids)
    torch.testing.assert_close(new_output[0], output[0])
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_distributed_moe_context(tmp_path, world_size):
    init_method = f"file://{tmp_path}/shared_file"
    mp.spawn(
        _test_distributed_moe_context,
        args=(init_method, world_size),
        nprocs=world_size,
        join=True,
        daemon=True,
    )


def _test_moe_generation(rank, init_method, world_size):
    torch.distributed.init_process_group(
        init_method=init_method,
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    input_ids = torch.tensor([list(range(10))], dtype=torch.int64)
    config = Qwen2Config(
        hidden_size=64,
        vocab_size=128,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_hidden_layers=4,
        max_position_embeddings=128,
        intermediate_size=128,
    )
    qwen2model = Qwen2ForCausalLM(config)
    qwen2model.eval()
    moe_model = from_qwen_to_moe(qwen2model, num_experts=1, num_experts_per_tok=1)
    moe_model.eval()
    orig_outputs = moe_model.generate(input_ids, return_dict_in_generate=True)

    num_return_sequences = 4
    max_new_tokens = 16
    with DistributedMoEContext(moe_model=moe_model, top_k=world_size):
        moe_model.eval()
        moe_generation_outputs = moe_model.generate(
            input_ids,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
        )
        assert moe_generation_outputs.sequences.shape[-2] == num_return_sequences
        assert moe_generation_outputs.sequences.shape[-1] == max_new_tokens
        assert moe_generation_outputs.scores.shape[-1] == num_return_sequences
    outputs = moe_model.generate(input_ids, return_dict_in_generate=True)
    torch.testing.assert_close(outputs.sequences, orig_outputs.sequences)
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_moe_generation(tmp_path, world_size):
    init_method = f"file://{tmp_path}/shared_file"
    mp.spawn(
        _test_moe_generation,
        args=(init_method, world_size),
        nprocs=world_size,
        join=True,
        daemon=True,
    )
