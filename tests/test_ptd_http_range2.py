"""Offline test for Range2 direct CPU tensor destination and strict 206 checks."""
import json, re, struct, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from patches.ptd.http_range2 import HTTPRange2TensorReader

class Handler(BaseHTTPRequestHandler):
    payload=b''
    seen=[]
    bad=False
    def log_message(self,*args):pass
    def do_GET(self):
        m=re.fullmatch(r'bytes=(\d+)-(\d+)',self.headers.get('Range',''))
        if self.bad or not m:
            self.send_response(200); self.send_header('Content-Length','0'); self.end_headers(); return
        a,b=map(int,m.groups());self.seen.append((a,b))
        self.send_response(206)
        self.send_header('Content-Range',f'bytes {a}-{b}/{len(self.payload)}')
        self.send_header('Content-Length',str(b-a+1));self.end_headers()
        for n in range(a,b+1,177):
            self.wfile.write(self.payload[n:min(n+177,b+1)])

class TestHTTPRange2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.tmp.name)/'fixture.safetensors'
        cls.values={'a':torch.arange(5003,dtype=torch.int32),
                    'b':torch.arange(1009,dtype=torch.float32),
                    'c':torch.arange(31999,dtype=torch.int8),
                    'd':torch.arange(5001,dtype=torch.float16)}
        raw=bytearray();header={}
        for key,v in cls.values.items():
            chunk=v.view(torch.uint8).numpy().tobytes()
            header[key]=dict(dtype={torch.int32:'I32',torch.float32:'F32',
                torch.int8:'I8',torch.float16:'F16'}[v.dtype],shape=list(v.shape),
                data_offsets=[len(raw),len(raw)+len(chunk)])
            raw.extend(chunk)
        h=json.dumps(header).encode()
        Handler.payload=struct.pack('<Q',len(h))+h+raw
        cls.path.write_bytes(Handler.payload)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.tmp.cleanup()
    def setUp(self):
        Handler.seen=[];Handler.bad=False;self.metrics=[]
        self.reader=HTTPRange2TensorReader(self.path,
            dataset_identity='pamcks/h3-ptd-v2-hop-00-20261103/versions/1',
            remote_filename='payload/fixture.safetensors',
            url_supplier=lambda *_:f'http://127.0.0.1:{self.server.server_port}/x',
            on_transfer=lambda *a:self.metrics.append(a))
    def test_sparse_batch(self):
        state=self.reader.read_tensors(('d','a','c'))
        for k,v in state.items():self.assertTrue(torch.equal(v,self.values[k]))
        self.assertEqual(len(Handler.seen),2)
        self.assertEqual(self.metrics[0][2],2)
        self.assertLess(self.metrics[0][0],self.metrics[0][1])
    def test_one_tensor(self):
        self.assertTrue(torch.equal(self.reader.read_tensor('a'),self.values['a']))
        self.assertEqual(len(Handler.seen),2)
    def test_rejects_http_200(self):
        Handler.bad=True
        with self.assertRaises(IOError):self.reader.read_tensors(('a','b'))
        self.assertFalse(self.metrics)
    def test_bank_telemetry_reports_both_throughput_definitions(self):
        from patches.ptd.bank_v2 import PTDBankV2
        bank=object.__new__(PTDBankV2)
        bank.http_range2_enabled=True
        bank.http_range2_debug=True
        bank._http2_lock=threading.Lock()
        bank._http2_stats=dict(logical_bytes=0,transferred_bytes=0,
            requests=0,transport_seconds=0.,tensor_ready_seconds=0.,batches=0)
        bank._http2_parts={};bank._http2_rates={}
        bank._http2_transfer('hop_00','aux_overlay.safetensors',6_000_000,
                             6_000_000,2,.03,.045)
        metrics=bank.http2_state()
        self.assertEqual(metrics['requests'],2)
        self.assertEqual(metrics['http_download_MBps'],200.)
        self.assertEqual(metrics['tensor_ready_MBps'],133.333)
        self.assertEqual(metrics['per_part']['hop_00']['overlays'],1)
        self.assertTrue(metrics['debug_enabled'])
        bank.http_range2_debug=False
        bank._http2_transfer('hop_00','aux_overlay.safetensors',6_000_000,
                             6_000_000,2,.03,.045)
        self.assertEqual(bank.http2_state()['requests'],2)

if __name__=='__main__':unittest.main()
