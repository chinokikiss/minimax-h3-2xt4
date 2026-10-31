"""Compatibility boundary for retired loader controls, without runtime branches."""
import warnings


def check_retired_loader_options(options):
    """Accept old Python calls; never reactivate removed algorithms."""
    constants = {'tp_quantization': 'scalemax', 'swiglu_reduction': 'native_int32',
                 'lora_concat_a': False, 'swiglu_padding': True, 'selective_b': True,
                 'checkpoint_read_chunk_mib': 0, 'scalar_score_workspace_mib': 96}
    unknown = options.keys() - constants.keys()
    if unknown:
        raise TypeError('Unknown H3 loader options: ' + ', '.join(sorted(unknown)))
    for name, value in options.items():
        if name in ('tp_quantization', 'swiglu_reduction', 'lora_concat_a') and value != constants[name]:
            raise ValueError(f'{name} is retired; the fixed value is {constants[name]!r}')
        if value != constants[name]:
            warnings.warn(f'H3 retired {name}={value!r}; using {constants[name]!r}',
                          FutureWarning, stacklevel=2)
