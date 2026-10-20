"""Small fixed matrix; no autotuning sweep or full-model targets."""
def configurations(component):
    tile,cache=(64,'group') if component=='video' else ((32,'q') if component=='dit' else (32,'group'))
    rows=[dict(name='current',baseline=True,tile=tile,cache=cache),
          dict(name='fused_prepare',fused=True,tile=tile,cache=cache)]
    if component=='video':
        rows += [dict(name='q128_row128',fused=True,tile=128,cache='group',threads=128,warp='row'),
                 dict(name='q128_row256',fused=True,tile=128,cache='group',threads=256,warp='row'),
                 dict(name='q128_qreuse256',fused=True,tile=128,cache='q',threads=256,warp='row'),
                 dict(name='kitchen',native=True)]
    elif component=='dit':
        rows += [dict(name='q64_row256',fused=True,tile=64,cache='q',threads=256,warp='row'),
                 dict(name='q64_square256',fused=True,tile=64,cache='q',threads=256,warp='square')]
    return rows
