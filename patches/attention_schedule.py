"""Per-sample DiT attention schedule parsing.

Syntax is a comma-separated list of 1-based NFE ranges:
"1:kitchen,2-7:sol@1.0,8:kitchen".

A Sol entry without "@tau" inherits RuntimeConfig.sol_tau.
Kitchen/native entries cannot carry a tau override.
"""
import math

_ALLOWED = frozenset(("kitchen", "sol", "native"))


def _normalize_entry(item):
    if not isinstance(item,(tuple,list)) or len(item) not in (3,4):
        raise ValueError('Each DiT attention schedule entry must be (start, end, backend[, tau])')
    start,end,backend=item[:3]
    tau=item[3] if len(item)==4 else None
    if type(start) is not int or start<1 or (end is not None and (type(end) is not int or end<start)):
        raise ValueError('NFE ranges are 1-based and nondecreasing')
    if backend not in _ALLOWED:
        raise ValueError('Unknown scheduled DiT attention backend: '+str(backend))
    if tau is not None:
        tau=float(tau)
        if backend!='sol':
            raise ValueError('Only Sol schedule entries may override tau')
        if not math.isfinite(tau) or tau<0:
            raise ValueError('Sol tau must be finite and nonnegative')
    return (start,end,backend,tau)


def validate_attention_schedule(schedule):
    result=[_normalize_entry(item) for item in schedule]
    for index,(start,end,_,_) in enumerate(result):
        upper=float('inf') if end is None else end
        for a,b,_,_ in result[:index]:
            b=float('inf') if b is None else b
            if max(start,a)<=min(upper,b):
                raise ValueError('DiT attention schedule ranges overlap')
    return tuple(result)


def parse_attention_schedule(value):
    if value is None:
        return ()
    if not isinstance(value, str):
        raise TypeError("dit_attention_schedule must be a string")
    value = value.strip()
    if not value:
        return ()
    result = []
    for raw in value.split(","):
        item = raw.strip()
        if not item or ":" not in item:
            raise ValueError("Invalid DiT attention schedule item: " + repr(raw))
        span, backend_spec = (part.strip() for part in item.split(":", 1))
        if "@" in backend_spec:
            backend,tau_text=(part.strip() for part in backend_spec.rsplit("@",1))
            if not tau_text:
                raise ValueError("Missing Sol tau override")
            tau=float(tau_text)
        else:
            backend,tau=backend_spec,None
        if backend not in _ALLOWED:
            raise ValueError("Unknown scheduled DiT attention backend: " + backend)
        if span.endswith("+"):
            start = int(span[:-1]); end = None
        elif "-" in span:
            first, last = span.split("-", 1); start, end = int(first), int(last)
        else:
            start = end = int(span)
        result.append((start,end,backend,tau))
    return validate_attention_schedule(result)


def schedule_uses_backend(schedule, backend):
    if backend not in _ALLOWED:
        raise ValueError("Unknown DiT attention backend: " + str(backend))
    return any(entry_backend == backend
               for _, _, entry_backend, _ in validate_attention_schedule(schedule))


def select_attention_policy(schedule, nfe, default, sol_tau):
    if type(nfe) is not int or nfe < 1:
        raise ValueError("NFE index must be a positive integer")
    if default not in _ALLOWED:
        raise ValueError("Unknown default DiT attention backend: " + str(default))
    sol_tau=float(sol_tau)
    if not math.isfinite(sol_tau) or sol_tau<0:
        raise ValueError("Sol tau must be finite and nonnegative")
    for start,end,backend,tau in validate_attention_schedule(schedule):
        if nfe>=start and (end is None or nfe<=end):
            return backend, (sol_tau if backend=='sol' and tau is None else tau)
    return default, (sol_tau if default=='sol' else None)


def select_attention(schedule, nfe, default):
    """Compatibility helper returning only the selected backend."""
    return select_attention_policy(schedule,nfe,default,1.0)[0]
