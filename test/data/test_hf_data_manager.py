import pytest

# Skip this entire test file
pytest.skip("Skipping all tests in test_hf_data_manager.py", allow_module_level=True)

from genrl.data.hf_data_manager import SerialHuggingFaceDataManager


def test_serial_huggingface_data_manager():
    data_manager = SerialHuggingFaceDataManager(
        path_or_name="roneneldan/TinyStories",
        tokenizer_path_or_name="gpt2",
        batch_size=4,
        tokenizer_kwargs={
            "pad_token_id": 0,
            "pad_token": "[PAD]",
        },
    )
    data_manager.initialize()

    _ = data_manager.train_batch()
    eval_data = data_manager.eval_data()
    for _ in eval_data:
        break

    text = "this is text"
    tokens = data_manager.encode(text)
    decoded_tokens = data_manager.decode(tokens)
    assert text == decoded_tokens
