import os
import pickle
import time
import logging
from typing import Any, Dict, List

import torch.distributed as dist
from hivemind import DHT, get_dht_time

from genrl.communication.communication import Communication
from genrl.serialization.game_tree import from_bytes, to_bytes

logger = logging.getLogger(__name__)

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
        logger.info(f"🔄 all_gather_object START: step={self.step_}, world_size={self.world_size}")

        try:
            # Проверка видимых адресов
            maddrs_start = time.monotonic()
            visible_maddrs = self.dht.get_visible_maddrs(latest=True)
            maddrs_time = time.monotonic() - maddrs_start
            logger.info(f"📡 get_visible_maddrs: {len(visible_maddrs)} addresses in {maddrs_time:.2f}s")
            logger.info(f"📍 Visible addresses: {visible_maddrs}")

            # Сериализация объекта
            serialize_start = time.monotonic()
            obj_bytes = to_bytes(obj)
            serialize_time = time.monotonic() - serialize_start
            logger.info(f"📦 Serialization took {serialize_time:.2f}s")

            # Сохранение в DHT
            store_start = time.monotonic()
            self.dht.store(
                key,
                subkey=str(self.dht.peer_id),
                value=obj_bytes,
                expiration_time=get_dht_time() + self.timeout,
                beam_size=self.beam_size,
            )
            store_time = time.monotonic() - store_start
            logger.info(f"💾 DHT store took {store_time:.2f}s")

            # Ожидание перед сбором
            logger.info("⏱️  Waiting 1s before gathering...")
            time.sleep(1)

            # Сбор результатов
            gather_start = time.monotonic()
            t_ = time.monotonic()
            iteration = 0

            while True:
                iteration += 1
                iter_start = time.monotonic()

                output_, _ = self.dht.get(key, beam_size=self.beam_size, latest=True)
                current_size = len(output_)

                iter_time = time.monotonic() - iter_start
                elapsed = time.monotonic() - t_

                logger.info(f"🔍 Iteration {iteration}: got {current_size}/{self.world_size} responses in {iter_time:.2f}s (total: {elapsed:.1f}s)")

                if current_size >= self.world_size:
                    logger.info(f"✅ Got all {self.world_size} responses!")
                    break
                else:
                    if elapsed > self.timeout:
                        logger.error(f"⏰ TIMEOUT after {elapsed:.1f}s! Got only {current_size}/{self.world_size}")
                        raise RuntimeError(
                            f"Failed to obtain {self.world_size} values for {key} within timeout."
                        )

                    # Логируем прогресс каждые 10 секунд
                    if iteration % 10 == 0:
                        logger.warning(f"⚠️  Still waiting: {current_size}/{self.world_size} after {elapsed:.1f}s")

            gather_time = time.monotonic() - gather_start
            logger.info(f"📊 Gathering took {gather_time:.2f}s in {iteration} iterations")

            self.step_ += 1

            # Обработка результатов
            process_start = time.monotonic()
            tmp = sorted(
                [(key, from_bytes(value.value)) for key, value in output_.items()],
                key=lambda x: x[0],
            )
            process_time = time.monotonic() - process_start
            logger.info(f"⚙️  Processing results took {process_time:.2f}s")

            total_time = time.monotonic() - start_time
            logger.info(f"✅ all_gather_object SUCCESS: total {total_time:.2f}s")

            return {key: value for key, value in tmp}

        except (BlockingIOError, EOFError) as e:
            total_time = time.monotonic() - start_time
            logger.error(f"❌ all_gather_object FAILED after {total_time:.2f}s: {e}")
            logger.info("🔄 Falling back to local object only")
            return {str(self.dht.peer_id): obj}

    def get_id(self):
        return str(self.dht.peer_id)
