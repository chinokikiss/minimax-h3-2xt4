"""ComfyUI lifecycle bridge only; GPU tensor ownership stays in ResidencyWindow."""
from contextlib import contextmanager
from pathlib import Path
import threading
import weakref
import sys

SERVICES=weakref.WeakSet()
LOCAL=threading.local()
BRIDGE=None
ORIGINAL_FREE=None


@contextmanager
def preserve_te(service):
    previous=getattr(LOCAL,'preserve',None);LOCAL.preserve=service
    try:yield
    finally:LOCAL.preserve=previous


def _compatible(service,prompt):
    for node in prompt.values():
        if node.get('class_type')=='MiniMaxH3TP2Loader':
            value=node.get('inputs',{}).get('text_encoder')
            if isinstance(value,str) and str(Path(value).resolve())==service.config.te_path:return True
    return False


class PromptBridge:
    def __init__(self,server):
        self.server=server;self.generation=0;self.timer=None;self.lock=threading.RLock()

    def invalidate(self):
        with self.lock:
            self.generation+=1
            if self.timer is not None:self.timer.cancel();self.timer=None

    def submitted(self,data):
        self.invalidate()
        return data

    def start(self,prompt_id):
        self.invalidate()
        running,_=self.server.prompt_queue.get_current_queue_volatile()
        prompt=next((item[2] for item in running if item[1]==prompt_id),{})
        for service in tuple(SERVICES):
            service.pause_idle(keep=_compatible(service,prompt))
            if hasattr(service,'configure_idle_vaes'):service.configure_idle_vaes(prompt)

    def complete(self,prompt_id,generation):
        # History is committed after the executor's end callback returns.
        with self.lock:
            if generation!=self.generation:return
            running,queued=self.server.prompt_queue.get_current_queue_volatile()
            if running or queued:return
            record=self.server.prompt_queue.get_history(prompt_id).get(prompt_id,{})
            status=record.get('status') or {}
            if status.get('completed') is not True or status.get('status_str')!='success':return
            prompt=record.get('prompt') or ()
            graph=prompt[2] if len(prompt)>2 else {}
        for service in tuple(SERVICES):
            with self.lock:
                if generation!=self.generation:return
            if _compatible(service,graph):
                if hasattr(service,'configure_idle_vaes'):service.configure_idle_vaes(graph)
                service.request_idle()

    def end(self,prompt_id):
        self.invalidate()
        with self.lock:
            self.timer=threading.Timer(.25,self.complete,args=(prompt_id,self.generation))
            self.timer.daemon=True;self.timer.start()


def register(service):
    global ORIGINAL_FREE,BRIDGE
    SERVICES.add(service)
    import comfy.model_management as management
    if ORIGINAL_FREE is None:
        ORIGINAL_FREE=management.free_memory
        def free_memory(memory_required,device,*args,**kwargs):
            result=ORIGINAL_FREE(memory_required,device,*args,**kwargs)
            if getattr(device,'type',None)!='cuda' or management.get_free_memory(device)>=memory_required:return result
            released=False
            for candidate in tuple(SERVICES):
                if candidate is not getattr(LOCAL,'preserve',None):
                    released=candidate.pause_idle(keep=False,pressure=True) or released
            if released:
                extra=ORIGINAL_FREE(memory_required,device,*args,**kwargs)
                if isinstance(result,list) and isinstance(extra,list):result+=extra
            return result
        management.free_memory=free_memory
    if BRIDGE is None:
        server=sys.modules.get('server')
        live=getattr(getattr(server,'PromptServer',None),'instance',None)
        if live is None:return  # Standalone runners use explicit request_idle.
        from comfy_execution.cache_provider import CacheProvider,register_cache_provider
        bridge=PromptBridge(live)
        class LifecycleProvider(CacheProvider):
            async def on_lookup(self,context):return None
            async def on_store(self,context,value):return None
            def should_cache(self,context,value=None):return False
            def on_prompt_start(self,prompt_id):bridge.start(prompt_id)
            def on_prompt_end(self,prompt_id):bridge.end(prompt_id)
        register_cache_provider(LifecycleProvider())
        live.add_on_prompt_handler(bridge.submitted);BRIDGE=bridge
