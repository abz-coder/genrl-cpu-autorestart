import os
import pickle
import time
from typing import Any, Dict, List

import torch.distributed as dist
from hivemind import DHT, get_dht_time

from genrl.communication.communication import Communication
from genrl.serialization.game_tree import from_bytes, to_bytes


class HivemindRendezvouz:
    _STORE = None
    _IS_MASTER = False
    _IS_LAMBDA = False

    @classmethod
    def init(cls, is_master: bool = False):
        cls._IS_MASTER = is_master
        cls._IS_LAMBDA = os.environ.get("LAMBDA", False)
        if cls._STORE is None and cls._IS_LAMBDA:
            world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
            cls._STORE = dist.TCPStore(
                host_name=os.environ["MASTER_ADDR"],
                port=int(os.environ["MASTER_PORT"]),
                is_master=is_master,
                world_size=world_size,
                wait_for_workers=True,
            )

    @classmethod
    def is_bootstrap(cls) -> bool:
        return cls._IS_MASTER

    @classmethod
    def set_initial_peers(cls, initial_peers):
        pass
        if cls._STORE is None and cls._IS_LAMBDA:
            cls.init()
        if cls._IS_LAMBDA:
            cls._STORE.set("initial_peers", pickle.dumps(initial_peers))

    @classmethod
    def get_initial_peers(cls):
        if cls._STORE is None and cls._IS_LAMBDA:
            cls.init()
        cls._STORE.wait(["initial_peers"])
        peer_bytes = cls._STORE.get("initial_peers")
        initial_peers = pickle.loads(peer_bytes)
        return initial_peers


class HivemindBackend(Communication):
    def __init__(
        self,
        initial_peers: List[str] | None = None,
        timeout: int = 600,
        disable_caching: bool = False,
        beam_size: int = 1000,
        **kwargs,
    ):
        self.world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        self.timeout = timeout
        self.bootstrap = HivemindRendezvouz.is_bootstrap()
        self.beam_size = beam_size
        self.dht = None

        if disable_caching:
            kwargs["cache_locally"] = False
            kwargs["cache_on_store"] = False

        if self.bootstrap:
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                initial_peers=initial_peers,
                **kwargs,
            )
            dht_maddrs = self.dht.get_visible_maddrs(latest=True)
            HivemindRendezvouz.set_initial_peers(dht_maddrs)
        else:
            initial_peers = initial_peers or HivemindRendezvouz.get_initial_peers()
            self.dht = DHT(
                start=True,
                host_maddrs=[f"/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                initial_peers=initial_peers,
                **kwargs,
            )
        self.step_ = 0

    def all_gather_object(self, obj: Any) -> Dict[str | int, Any]:
        start_time = time.monotonic()
        key = str(self.step_)
        try:
            visible_maddrs = self.dht.get_visible_maddrs(latest=True)
            print(f"📍 Visible addresses: {visible_maddrs}")
            
            obj_bytes = to_bytes(obj)
            store_start = time.monotonic()
            self.dht.store(
                key,
                subkey=str(self.dht.peer_id),
                value=obj_bytes,
                expiration_time=get_dht_time() + self.timeout,
                beam_size=self.beam_size,  
            )
            store_time = time.monotonic() - store_start
            print(f"💾 DHT store took {store_time:.2f}s")
            
            time.sleep(1)
            t_ = time.monotonic()
            while True:
                output_, _ = self.dht.get(key, beam_size=self.beam_size, latest=True)
                current_size = len(output_)
                print(f"🔍 Got {current_size} responses")
                
                if current_size >= self.world_size:
                    break
                else:
                    if time.monotonic() - t_ > self.timeout:
                        raise RuntimeError(
                            f"Failed to obtain {self.world_size} values for {key} within timeout."
                        )
            self.step_ += 1

            tmp = sorted(
                [(key, from_bytes(value.value)) for key, value in output_.items()],
                key=lambda x: x[0],
            )
            
            total_time = time.monotonic() - start_time
            print(f"✅ all_gather_object completed in {total_time:.2f}s")
            
            return {key: value for key, value in tmp}
        except (BlockingIOError, EOFError) as e:
            total_time = time.monotonic() - start_time
            print(f"❌ all_gather_object failed after {total_time:.2f}s: {e}")
            return {str(self.dht.peer_id): obj}

    def get_id(self):
        return str(self.dht.peer_id)
