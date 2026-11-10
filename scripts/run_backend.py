"""Localhost-only single-worker synthetic API, with native thread caps."""
import os
from pathlib import Path
import sys


def main():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS'):
        os.environ[key]='4'
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    import uvicorn
    # No reload/multiple workers: inference is serialized within one service.
    uvicorn.run('trafficpulse.api.app:create_app',factory=True,host='127.0.0.1',port=8000,workers=1)


if __name__=='__main__':main()
