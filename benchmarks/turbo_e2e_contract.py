"""Pure metadata checks for the requested Turbo/base compatibility."""

def validate_header(header, native):
    known, targets = set(), []
    for name in header:
        if not name.endswith('.lora_A.weight'):
            continue
        prefix = name.removesuffix('.lora_A.weight')
        target = prefix.removeprefix('diffusion_model.')
        b, alpha = prefix + '.lora_B.weight', prefix + '.alpha'
        a_shape = header[name]['shape']
        b_shape = header[b]['shape']
        expected = native[target + '.weight']['shape']
        if (len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]
                or expected != [b_shape[0], a_shape[1]]):
            raise ValueError('Turbo/base geometry mismatch: ' + target)
        known.update((name, b))
        if alpha in header:
            if header[alpha]['shape'] not in ([], [1]):
                raise ValueError('Invalid Turbo alpha shape')
            known.add(alpha)
        targets.append(target)
    if set(header) - known - {'__metadata__'} or not targets:
        raise ValueError('Unconsumed Turbo keys')
    return targets

