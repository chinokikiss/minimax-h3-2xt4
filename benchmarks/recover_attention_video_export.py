"""Recover only a failed video export after successful native sampling/decode."""
import json
from pathlib import Path
import torch
import av

if __name__=='__main__':
    directory=Path('D:/attention_eval/tokyo_ref2va_20261013')
    state=json.loads((directory/'run_state.json').read_text())
    assert 'Expected numpy array' in state['traceback']
    assert len(list(directory.glob('denoiser_call*.pt')))==20
    manifest=json.loads((directory/'capture_manifest.json').read_text())
    assert len(manifest['records'])==25
    pixels=torch.load(directory/'native_pixels.pt',map_location='cpu',weights_only=True)
    frames=pixels[0] if pixels.ndim==5 and pixels.shape[0]==1 else pixels
    assert tuple(frames.shape)==(90,352,640,3),frames.shape
    with av.open(str(directory/'native_video.mp4'),'w') as container:
        stream=container.add_stream('libx264',rate=24);stream.width=640;stream.height=352;stream.pix_fmt='yuv420p'
        for frame in frames:
            array=frame.clamp(0,1).mul(255).round().byte().numpy()
            for packet in stream.encode(av.VideoFrame.from_ndarray(array,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    (directory/'video_export_failure.json').write_text(json.dumps(state,indent=2))
    state.pop('traceback');state.update(status='complete',captures=25,export_recovered_without_model_rerun=True)
    (directory/'run_state.json').write_text(json.dumps(state,indent=2))
    print(frames.shape,'native capture complete')
