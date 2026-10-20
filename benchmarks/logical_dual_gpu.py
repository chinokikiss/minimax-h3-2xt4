"""Execute two real CUDA ranks on cuda:0, substituting only their transport."""
from concurrent.futures import ThreadPoolExecutor
import threading
import torch


def isolate_cublas_workspaces():
    """Emulate device-private scratch storage for ranks sharing cuda:0."""
    import comfy_kitchen as ck
    backend = ck.backends.cuda
    workspaces = {}
    def workspace():
        device = torch.cuda.current_device()
        key = (device, torch.cuda.current_stream(device).cuda_stream)
        if key not in workspaces:
            workspaces[key] = torch.empty(backend.get_cublas_workspace_size_bytes(),
                                          dtype=torch.uint8, device=device)
        return workspaces[key]
    backend.get_cublas_workspace = workspace


class LogicalDualGPU:
    ReduceOp = torch.distributed.ReduceOp
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.local = threading.local()
        self.condition = threading.Condition()
        self.steps = {}
        self.failure = None
        self.streams = [torch.cuda.Stream(device=0) for _ in range(2)]

    def is_initialized(self):
        return True

    def get_rank(self):
        return self.local.rank

    def get_world_size(self):
        return 2

    def _exchange(self, kind, tensor, output, op=None):
        torch.cuda.current_stream().synchronize()
        rank, step = self.local.rank, self.local.step
        self.local.step += 1
        snapshot = tensor.clone()
        torch.cuda.current_stream().synchronize()
        with self.condition:
            entry = self.steps.setdefault(step, {})
            entry[rank] = (kind, snapshot, output, op)
            if len(entry) == 2:
                try:
                    a, b = entry[0], entry[1]
                    if a[0] != b[0] or a[3] != b[3]:
                        raise RuntimeError(f"Collective order mismatch at step {step}")
                    with torch.cuda.device(0), torch.inference_mode():
                        if kind == "all_reduce":
                            if op == self.ReduceOp.MAX:
                                result = torch.maximum(a[1], b[1])
                            elif op == self.ReduceOp.MIN:
                                result = torch.minimum(a[1], b[1])
                            else:
                                result = a[1] + b[1]
                            a[2].copy_(result); b[2].copy_(result)
                        elif kind == "broadcast":
                            for record in [a,b]:record[2].copy_(entry[op][1])
                        elif kind == "all_gather":
                            for record in [a, b]:
                                for i in range(2):
                                    record[2][i].copy_(entry[i][1])
                        elif kind == "all_gather_into_tensor":
                            result = torch.cat([a[1], b[1]], dim=0)
                            for record in [a, b]:
                                record[2].copy_(result.reshape(record[2].shape))
                        elif kind == "reduce_scatter_tensor":
                            result = a[1] + b[1]
                            for i in range(2):
                                entry[i][2].copy_(result.chunk(2, dim=0)[i])
                        elif kind == "reduce_scatter":
                            for i in range(2):
                                entry[i][2].copy_(a[1][i] + b[1][i])
                        else:
                            raise RuntimeError(kind)
                        torch.cuda.current_stream().synchronize()
                    entry["done"] = True
                except Exception as error:
                    self.failure = error
                self.condition.notify_all()
            ready = self.condition.wait_for(lambda: entry.get("done") or self.failure is not None, self.timeout)
            if not ready:
                self.failure = RuntimeError(f"Logical rank did not reach collective {step}: {kind}")
                self.condition.notify_all()
            if self.failure is not None:
                raise self.failure
            entry['returned'] = entry.get('returned', 0) + 1
            if entry['returned'] == 2:
                del self.steps[step]

    def all_reduce(self, tensor, op=None):
        return self._exchange("all_reduce", tensor, tensor, op)

    def broadcast(self,tensor,src=0):
        return self._exchange('broadcast',tensor,tensor,src)

    def all_gather(self, output, tensor):
        return self._exchange("all_gather", tensor, output)

    def all_gather_into_tensor(self, output, tensor):
        return self._exchange("all_gather_into_tensor", tensor, output)

    def reduce_scatter_tensor(self, output, tensor, op=None):
        return self._exchange("reduce_scatter_tensor", tensor, output, op)

    def reduce_scatter(self, output, tensors, op=None):
        return self._exchange("reduce_scatter", torch.stack(tensors), output, op)

    def run(self, calls):
        self.steps, self.failure = {}, None
        torch.cuda.synchronize(0)
        def worker(rank):
            self.local.rank, self.local.step = rank, 0
            try:
                with torch.cuda.device(0), torch.cuda.stream(self.streams[rank]), torch.inference_mode():
                    out = calls[rank]()
                    torch.cuda.current_stream().synchronize()
                    return out
            except Exception as error:
                with self.condition:
                    self.failure = error
                    self.condition.notify_all()
                raise
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, r) for r in range(2)]
            return [f.result() for f in futures]
