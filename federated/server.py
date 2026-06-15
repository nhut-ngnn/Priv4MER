import os
import sys


_SERVER_DIR = os.path.join(os.path.dirname(__file__), "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

import server_fedavg
import server_fedprox


def _select_server(cfg):
    method = str(cfg.get("fl_method", "fedavg")).lower()
    if method == "fedprox":
        return server_fedprox
    return server_fedavg


def run_stage(*args, **kwargs):
    cfg = kwargs.get("cfg")
    if cfg is None and len(args) >= 3:
        cfg = args[2]
    module = _select_server(cfg or {})
    return module.run_stage(*args, **kwargs)
