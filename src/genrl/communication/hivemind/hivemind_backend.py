import os
import pickle
import time
import logging
from typing import Any, Dict, List, Optional

import torch.distributed as dist
from hivemind import DHT, get_dht_time

from genrl.communication.communication import Communication
from genrl.serialization.game_tree import from_bytes, to_bytes

logger = logging.getLogger(__name__)

class HivemindRendezvouz:
    _STORE = None
    _IS_MASTER = False
    _IS_LAMBDA = False
    _initial_peers: List[str] = [
        # Можно добавить сюда дефолтных бустрэп пиров, если хочешь
        # "/ip4/38.101.215.15/tcp/30011/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ",
        # "/ip4/38.101.215.15/tcp/30012/p2p/QmWhiaLrx3HRZfgXc2i7KW5nMUNK7P9tRc71yFJdGEZKkC",
        # "/ip4/38.101.215.15/tcp/30013/p2p/QmQa1SCfYTxx7RvU7qJJRo79Zm1RAwPpkeLueDVJuBBmFp"
    ]

    @classmethod
    def init(cls, is_master: bool = False):
        cls._IS_MASTER = is_master
        cls._IS_LAMBDA = os.environ.get("LAMBDA", False)
        logger.info(f"?? HivemindRendezvouz.init: is_master={is_master}, is_lambda={cls._IS_LAMBDA}")
        if cls._STORE is None and cls._IS_LAMBDA:
            world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
            logger.info(f"?? Initializing TCPStore with world_size={world_size}")
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
    def set_initial_peers(cls, initial_peers: List[str]):
        logger.info(f"?? Setting initial peers: {initial_peers}")
        if cls._STORE is None and cls._IS_LAMBDA:
            cls.init()
        if cls._IS_LAMBDA and cls._STORE is not None:
            cls._STORE.set("initial_peers", pickle.dumps(initial_peers))
            logger.info("? Initial peers stored in TCPStore")

    @classmethod
    def get_initial_peers(cls) -> List[str]:
        logger.info("?? Getting initial peers...")
        
        # Если хочешь отключить lookup из цепочки (например, для локального запуска)
        if not getattr(cls, 'force_chain_lookup', True):
            logger.info("??  force_chain_lookup=False, returning empty initial peers list")
            return []

        # Фильтрация "мертвых" пиров по IP 38.101.215.15
        dead_ip_prefix = "/ip4/38.101.215.15"
        logger.info(f"?? Filtering dead peers with prefix: {dead_ip_prefix}")

        # Получаем пиров из store, если в режиме lambda, иначе из _initial_peers
        if cls._STORE is not None and cls._IS_LAMBDA:
            logger.info("?? Getting peers from TCPStore...")
            cls._STORE.wait(["initial_peers"])
            peer_bytes = cls._STORE.get("initial_peers")
            if peer_bytes is not None:
                peers = pickle.loads(peer_bytes)
                logger.info(f"?? Loaded {len(peers)} peers from store: {peers}")
            else:
                peers = []
                logger.warning("??  No peers found in TCPStore")
        else:
            peers = cls._initial_peers
            logger.info(f"?? Using default peers: {peers}")

        alive_peers = [p for p in peers if not p.startswith(dead_ip_prefix)]
        filtered_count = len(peers) - len(alive_peers)
        
        if filtered_count > 0:
            logger.info(f"?? Filtered out {filtered_count} dead peers")

        if alive_peers:
            logger.info(f"? Returning {len(alive_peers)} alive initial peers: {alive_peers}")
            return alive_peers
        else:
            logger.warning("??  No alive initial peers found, returning empty list")
            logger.info("?? System will rely on blockchain bootnodes for network discovery")
            return []

class HivemindBackend(Communication):
    def __init__(
        self,
        initial_peers: Optional[List[str]] = None,
        timeout: int = 600,
        disable_caching: bool = False,
        beam_size: int = 1000,
        **kwargs,
    ):
        logger.info("?? Initializing HivemindBackend-optimized-with-initial-peers...")
        logger.info(f"?? Constructor initial_peers parameter: {initial_peers}")
        
        self.world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        self.timeout = timeout
        self.bootstrap = HivemindRendezvouz.is_bootstrap()
        self.beam_size = beam_size
        self.dht = None

        logger.info(f"?? Configuration: world_size={self.world_size}, timeout={timeout}, bootstrap={self.bootstrap}")

        if disable_caching:
            kwargs['cache_locally'] = False
            kwargs['cache_on_store'] = False
            logger.info("?? Caching disabled")

        # Обработка initial_peers как в оригинальном коде
        if self.bootstrap:
            # Bootstrap нода использует переданные initial_peers (могут быть None/пустые)
            logger.info("???  Starting as BOOTSTRAP node...")
            if initial_peers:
                logger.info(f"?? Bootstrap using provided initial_peers: {initial_peers}")
            else:
                logger.info("?? Bootstrap starting with no initial_peers (standalone)")
            
            # Применяем фильтрацию мертвых пиров для bootstrap тоже
            if initial_peers:
                dead_ip_prefix = "/ip4/38.101.215.15"
                alive_peers = [p for p in initial_peers if not p.startswith(dead_ip_prefix)]
                filtered_count = len(initial_peers) - len(alive_peers)
                if filtered_count > 0:
                    logger.info(f"?? Bootstrap: Filtered out {filtered_count} dead peers")
                    initial_peers = alive_peers
        else:
            # Participant нода: используем переданные пиры ИЛИ получаем из HivemindRendezvouz
            logger.info("?? Starting as PARTICIPANT node...")
            logger.info(f"?? Constructor initial_peers: {initial_peers}")
            
            if initial_peers is None:
                logger.warning("??  initial_peers is None, will call get_initial_peers()")
                try:
                    initial_peers = HivemindRendezvouz.get_initial_peers()
                    logger.info(f"? Retrieved initial_peers from HivemindRendezvouz: {initial_peers}")
                except Exception as e:
                    logger.error(f"?? Error getting initial_peers from HivemindRendezvouz: {e}")
                    logger.info("?? Falling back to empty list - will rely on blockchain coordination")
                    initial_peers = []
            else:
                logger.info(f"?? Using provided initial_peers: {initial_peers}")
                # Применяем фильтрацию мертвых пиров
                dead_ip_prefix = "/ip4/38.101.215.15"
                alive_peers = [p for p in initial_peers if not p.startswith(dead_ip_prefix)]
                filtered_count = len(initial_peers) - len(alive_peers)
                if filtered_count > 0:
                    logger.info(f"?? Participant: Filtered out {filtered_count} dead peers")
                    initial_peers = alive_peers

        logger.info(f"?? Final initial_peers for DHT: {initial_peers}")

        if self.bootstrap:
            # Bootstrap нода — запускает DHT с заданными initial_peers (можно пустой список)
            logger.info("???  Creating Bootstrap DHT...")
            self.dht = DHT(
                start=True,
                host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                initial_peers=initial_peers,
                **kwargs,
            )
            dht_maddrs = self.dht.get_visible_maddrs(latest=True)
            HivemindRendezvouz.set_initial_peers(dht_maddrs)
            logger.info(f"? Bootstrap DHT started successfully!")
            logger.info(f"?? Bootstrap node visible addresses: {dht_maddrs}")
        else:
            # Участник сети — подключается к bootstrap пирами
            logger.info("?? Creating Participant DHT...")
            
            if not initial_peers:
                logger.warning("??  Starting DHT with empty initial_peers!")
                logger.info("?? Will rely on blockchain coordinator for network discovery")
                logger.info("?? This is NORMAL - system will connect to main network via blockchain bootnodes")
            
            self.dht = DHT(
                start=True,
                host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                initial_peers=initial_peers,
                **kwargs,
            )
            logger.info(f"? Participant DHT started successfully!")
            logger.info(f"?? Connected with initial peers: {initial_peers}")

        # Проверяем, что DHT успешно инициализирован
        if self.dht is None:
            logger.error("? DHT initialization failed!")
            raise RuntimeError("DHT initialization failed")
        
        # Получаем и логируем peer ID
        self.peer_id = str(self.dht.peer_id)
        logger.info(f"?? Generated Peer ID: {self.peer_id}")
        
        # Получаем видимые адреса для проверки сетевой связности
        try:
            visible_maddrs = self.dht.get_visible_maddrs(latest=True)
            logger.info(f"?? DHT visible addresses: {visible_maddrs}")
            
            if visible_maddrs:
                logger.info("? DHT is accessible from the network!")
            else:
                logger.warning("??  DHT has no visible addresses - possible network issues")
                
        except Exception as e:
            logger.error(f"? Error getting visible addresses: {e}")

        # Проверяем подключение к другим пирам
        try:
            # Попытаемся получить информацию о ближайших пирах
            logger.info("?? Checking network connectivity...")
            
            # Даем время DHT для обнаружения сети
            time.sleep(2)
            
            # Проверим, есть ли у нас подключения к другим пирам
            # Это работает только если у нас есть другие пиры в сети
            routing_table_size = len(self.dht.get_visible_maddrs(latest=True))
            logger.info(f"?? DHT routing table size: {routing_table_size}")
            
            if routing_table_size > 0:
                logger.info("? DHT has network connections - connected to swarm!")
            else:
                logger.info("??  DHT starting in isolated mode - will connect via blockchain coordination")
                
        except Exception as e:
            logger.warning(f"??  Could not check network connectivity: {e}")

        logger.info("?? HivemindBackend initialization complete!")
        logger.info("?? Next steps: SwarmCoordinator will get bootnodes from blockchain and register this peer")

        self.step_ = 0

    def all_gather_object(self, obj: Any) -> Dict[str | int, Any]:
        assert self.dht is not None, "DHT must be initialized before calling all_gather_object"
        
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
            if self.dht is not None:
                peer_id = str(self.dht.peer_id)
            else:
                peer_id = "unknown"
            return {peer_id: obj}

    def get_id(self):
        return str(self.dht.peer_id)
