"""Count attention inside completed denoiser commands, including older traces."""


def denoiser_attention_counts(benchmark, kind='kitchen/int8_attention'):
    rows = benchmark['gpu']
    commands = [row['nfe'] for row in benchmark['commands'] if row['command'] == 'denoise']
    if not commands or len(commands) != len(set(commands)):
        raise ValueError('Expected unique denoiser command records')
    ends = {}
    for index, row in enumerate(rows):
        if row.get('kind') == 'command_gpu_span' and row.get('command') == 'denoise':
            nfe = row['nfe']
            if nfe in ends:
                raise ValueError('Duplicate denoiser completion marker')
            ends[nfe] = index
    if set(ends) != set(commands):
        raise ValueError('Missing or unexpected denoiser completion marker')
    ordered_ends = [ends[nfe] for nfe in commands]
    if ordered_ends != sorted(ordered_ends):
        raise ValueError('Denoiser markers do not match command order')
    starts = dict(zip(commands, [-1] + ordered_ends[:-1]))
    inside = dict.fromkeys(commands, 0)
    outside = dict.fromkeys(commands, 0)
    for index, row in enumerate(rows):
        if row.get('kind') != kind or row.get('command') != 'denoise':
            continue
        nfe = row['nfe']
        if nfe not in ends:
            raise ValueError('Attention event has no matching denoiser command')
        target = inside if starts[nfe] < index < ends[nfe] else outside
        target[nfe] += 1
    return dict(total=sum(inside.values()), per_step=inside,
                outside_command=sum(outside.values()), outside_per_step=outside)


def kitchen_denoiser_calls(benchmark):
    return denoiser_attention_counts(benchmark)['total']
