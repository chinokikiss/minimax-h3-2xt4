"""Decoder-only physical worker harness using the production dispatch/IPC path."""
import atexit
from datetime import timedelta
import os
import threading
import time
import traceback
from types import SimpleNamespace
import weakref

import torch
from patches.comfy_tp_runtime import PhysicalTPService, RankRuntime, RuntimeConfig, worker_reply
from patches.host_weight_cache import HostBudget, HostWeightCache, GiB


class CountComm:
    def __init__(self, dist):
        self.dist, self.ReduceOp = dist, dist.ReduceOp
        self.counts = {}

    def _count(self, name, value):
        item = self.counts.setdefault(name, dict(calls=0, operand_bytes=0))
        item['calls'] += 1
        item['operand_bytes'] += value.numel() * value.element_size()

    def all_reduce(self, value, **kwargs):
        self._count('all_reduce', value)
        return self.dist.all_reduce(value, **kwargs)

    def broadcast(self, value, **kwargs):
        self._count('broadcast', value)
        return self.dist.broadcast(value, **kwargs)

    def all_gather_into_tensor(self, output, value):
        self._count('all_gather_into_tensor', value)
        return self.dist.all_gather_into_tensor(output, value)


def decoder_worker(rank, config, budget, port, connection):
    import torch.distributed as dist
    runtime = None
    try:
        os.environ.update(NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_MIN_NCHANNELS='4', NCCL_MAX_NCHANNELS='4')
        os.environ.pop('NCCL_PROTO', None)
        torch.cuda.set_device(rank)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        from patches.optimized_tp import enable
        enable()
        store = dist.TCPStore('127.0.0.1', port, 3, False, timedelta(seconds=config.timeout_seconds))
        dist.init_process_group('nccl', store=store, rank=rank, world_size=2,
                                timeout=timedelta(seconds=config.timeout_seconds))
        runtime = RankRuntime.__new__(RankRuntime)
        runtime.config, runtime.rank, runtime.device = config, rank, torch.device('cuda', rank)
        runtime.comm = CountComm(dist)
        runtime.readers = {'dit': SimpleNamespace(), 'te': SimpleNamespace()}
        runtime.cache = HostWeightCache(budget, 256 * 2**20)
        runtime.startup_cache = HostWeightCache(budget, 256 * 2**20)
        runtime.model = runtime.window = runtime.preprocessing = runtime.vae_model = runtime.idle_window = None
        runtime.kind = runtime.vae_kind = None
        runtime.aux_reservation = runtime.calls = 0
        runtime.sample_active = runtime.completed_decode = runtime.decode_lookahead_started = False
        runtime.vae_numeric_flags, runtime.events = None, []
        rows = []
        connection.send(('ready', None))
        while True:
            sequence, command, args, kwargs = connection.recv()
            if command == 'stats':
                connection.send((sequence, dict(rank=rank, commands=rows)))
                continue
            wall, cpu = time.monotonic_ns(), time.thread_time_ns()
            decode = command in ('video_pixels', 'video_temporal_pair')
            runtime.comm.counts = {}
            if command == 'vae_prepare':
                torch.cuda.reset_peak_memory_stats(rank)
            if decode:
                tile_calls = []
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            result = runtime.dispatch(command, args, kwargs)
            attention = None
            if command == 'vae_prepare' and result.get('accepted'):
                from patches.parallel_vae import query_partitioning_status
                attention = query_partitioning_status(runtime.vae_model)
                planner = getattr(runtime, 'vae_temporal_planner', None)
                if planner is not None:
                    original_decode = planner.decode_pixels
                    def observe_tiles(z):
                        tile = dict(shape=list(z.shape), success=False)
                        tile_calls.append(tile)
                        try:
                            value = original_decode(z)
                            tile['success'] = True
                            return value
                        except torch.cuda.OutOfMemoryError:
                            tile['oom'] = True
                            raise
                    planner.decode_pixels = observe_tiles
            if command == 'vae_prepare':
                tile_calls = []
            if decode:
                end.record()
                end.synchronize()
            reply = worker_reply(result, rank, command)
            torch.cuda.synchronize(rank)
            rows.append(dict(command=command, start_ns=wall, end_ns=time.monotonic_ns(),
                thread_cpu_ns=time.thread_time_ns()-cpu, gpu_span_ms=start.elapsed_time(end) if decode else None,
                reply_bytes=reply.numel()*reply.element_size() if isinstance(reply, torch.Tensor) else 0,
                collectives=runtime.comm.counts, peak_allocated=torch.cuda.max_memory_allocated(rank),
                peak_reserved=torch.cuda.max_memory_reserved(rank), attention=attention,
                native_tiles=list(tile_calls) if decode else [], runtime_events=list(runtime.events)))
            connection.send((sequence, reply))
            if command == 'close':
                break
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if runtime is not None:
            runtime.cache.close()
            runtime.startup_cache.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


class DecoderService(PhysicalTPService):
    """No TE/DiT objects. Keep replicas resident only within a benchmark phase."""
    def __init__(self, *, host_bytes=16 * GiB, timeout_seconds=600):
        import torch.multiprocessing as mp
        import torch.distributed as dist
        self.config = RuntimeConfig('unused', 'unused', cross_request_te=False,
                                    prefetch_lifecycle=False, timeout_seconds=timeout_seconds,
                                    host_bytes=host_bytes)
        self.lock, self.sequence, self.closed = threading.RLock(), 0, False
        self.idle_held, self.idle_vae_thread = False, None
        self.idle_vae_names, self.idle_vae_cancel = None, threading.Event()
        self.idle_vae_stats = dict(loads=0, errors=0, last_error=None)
        self.staged_vaes = weakref.WeakSet()
        self.resident_mode = None
        self.parent_commands = []
        context = mp.get_context('spawn')
        self.budget = HostBudget(host_bytes, context=context, pressure_aware=True)
        self.store = dist.TCPStore('127.0.0.1', 0, 3, True,
                                   timedelta(seconds=timeout_seconds), wait_for_workers=False)
        self.connections, self.processes = [], []
        try:
            for rank in (0, 1):
                parent, child = context.Pipe()
                process = context.Process(target=decoder_worker,
                    args=(rank, self.config, self.budget, self.store.port, child), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                assert self._receive(connection)[0] == 'ready'
        except BaseException:
            self.abort()
            raise
        atexit.register(self.close)

    def call(self, command, *args, **kwargs):
        begin = time.monotonic_ns()
        if command == 'vae_release':
            # Benchmark-only: do not mix repeated weight I/O into steady state.
            return None
        if command == 'vae_prepare':
            mode = args[4] if len(args) > 4 else 'query'
            if self.resident_mode == mode:
                return self.prepared
            self.release()
            result = super().call(command, *args, **kwargs)
            if result['accepted']:
                self.resident_mode, self.prepared = mode, result
        else:
            result = super().call(command, *args, **kwargs)
        self.parent_commands.append(dict(command=command, start_ns=begin, end_ns=time.monotonic_ns()))
        return result

    def release(self):
        if self.resident_mode is not None and not self.closed:
            super().call('vae_release')
        self.resident_mode = None
