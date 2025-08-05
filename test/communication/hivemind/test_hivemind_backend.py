import os

import pytest

# Skip this entire test file
pytest.skip("Skipping all tests in test_hivemind_backend.py", allow_module_level=True)
import torch.multiprocessing as mp

from genrl.communication.hivemind.hivemind_backend import (
    HivemindBackend,
    HivemindRendezvouz,
)


def _test_hivemind_backend(rank, world_size):
    HivemindRendezvouz.init(is_master=rank == 0)

    backend = HivemindBackend(timeout=5)
    obj = [rank]
    gathered_obj = backend.all_gather_object(obj)
    print(gathered_obj)
    assert list(sorted(gathered_obj.values(), key=lambda x: x[0])) == [
        [i] for i in range(world_size)
    ]
    assert gathered_obj[backend.get_id()] == [rank]


@pytest.mark.parametrize("world_size", [1, 2])
def test_hivemind_backend(world_size):
    os.environ["HIVEMIND_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29400"
    mp.spawn(
        _test_hivemind_backend,
        args=(world_size,),
        nprocs=world_size,
        join=True,
        daemon=False,
    )
