"""Real ComfyUI callback contracts using deterministic queue/history fixtures."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import patches.comfy_idle_lifecycle as lifecycle


class Service:
    def __init__(self):self.config=SimpleNamespace(te_path=str(Path('te.safetensors').resolve()));self.starts=0;self.stops=[]
    def request_idle(self):self.starts+=1
    def pause_idle(self,**kw):self.stops.append(kw)


class Lifecycle(unittest.TestCase):
    def setup_bridge(self,status=True,queued=False):
        service=Service();graph={'1':{'class_type':'MiniMaxH3TP2Loader','inputs':{'text_encoder':service.config.te_path}}}
        record=dict(prompt=(0,'request',graph),status=dict(completed=status,status_str='success' if status else 'error'))
        queue=SimpleNamespace(get_current_queue_volatile=lambda:([],[('next',)] if queued else []),
            get_history=lambda identity:{identity:record})
        bridge=lifecycle.PromptBridge(SimpleNamespace(prompt_queue=queue))
        return service,bridge,record

    def test_only_successful_idle_and_matching_model_can_start(self):
        service,bridge,record=self.setup_bridge()
        with patch.object(lifecycle,'SERVICES',{service}):
            bridge.complete('request',0);self.assertEqual(service.starts,1)
            record['status']['completed']=False;bridge.complete('request',0);self.assertEqual(service.starts,1)
            record['status']['completed']=True;record['prompt']=(0,'request',{})
            bridge.complete('request',0);self.assertEqual(service.starts,1)

    def test_queued_and_rapidly_submitted_requests_suppress_prefetch(self):
        service,bridge,_=self.setup_bridge(queued=True)
        with patch.object(lifecycle,'SERVICES',{service}):bridge.complete('request',0)
        self.assertEqual(service.starts,0)
        service,bridge,_=self.setup_bridge()
        with patch.object(lifecycle,'SERVICES',{service}):
            bridge.submitted({'prompt':{}});bridge.complete('request',0)
        self.assertEqual(service.starts,0)

    def test_switch_discards_previous_te_and_same_model_keeps_entries(self):
        service,bridge,record=self.setup_bridge()
        bridge.server.prompt_queue.get_current_queue_volatile=lambda:([record['prompt']],[])
        with patch.object(lifecycle,'SERVICES',{service}):
            bridge.start('request');self.assertTrue(service.stops[-1]['keep'])
            record['prompt'][2]['1']['inputs']['text_encoder']='other.safetensors'
            bridge.start('request');self.assertFalse(service.stops[-1]['keep'])


if __name__=='__main__':unittest.main()
