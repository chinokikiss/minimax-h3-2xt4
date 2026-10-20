"""Process-scoped TLS1.2 workaround for local TLS1.3 EOF; verify certificates."""
import runpy
import ssl
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter


class TLS12Adapter(HTTPAdapter):
    def init_poolmanager(self,*args,**kwargs):
        context=ssl.create_default_context()
        context.minimum_version=ssl.TLSVersion.TLSv1_2
        context.maximum_version=ssl.TLSVersion.TLSv1_2
        kwargs['ssl_context']=context
        return super().init_poolmanager(*args,**kwargs)


original_init=requests.Session.__init__
def session_init(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    self.mount('https://',TLS12Adapter())
requests.Session.__init__=session_init

# SDK 2.2.4 accepts an expired cached OAuth token for another 30 minutes.
# Refresh through its normal credential mechanism before expiry instead.
from kagglesdk.kaggle_creds import KaggleCredentials
def token_needs_refresh(self):
    return not self._access_token_expiration or self._access_token_expiration <= datetime.now(timezone.utc)+timedelta(minutes=2)
KaggleCredentials.access_token_has_expired=token_needs_refresh


if __name__=='__main__':
    target=sys.argv[1];sys.argv=sys.argv[1:]
    if target=='--cli':
        sys.argv[0]='kaggle'
        from kaggle.cli import main
        main()
    else:
        sys.path.insert(0,str(Path(target).resolve().parent))
        runpy.run_path(target,run_name='__main__')
