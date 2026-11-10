"""FastAPI factory for a local synthetic demonstration; no training endpoints."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI,HTTPException,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import HealthResponse,MetadataResponse,ReplayRequest,ReplayResponse
from .service import ReplayInputError,ReplayService


def create_app(service_factory=None):
    factory=service_factory or (lambda:ReplayService(Path(__file__).resolve().parents[3]))

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.replay=factory()
        except Exception:
            # Do not echo artifact paths, native library exceptions or environment.
            app.state.replay=None
        yield
        app.state.replay=None

    app=FastAPI(title='TrafficPulse synthetic replay',version='1.1',debug=False,lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request,exc):
        return JSONResponse(status_code=422,content={'detail':[
            {'field':'.'.join(str(x) for x in e['loc']),'type':e['type'],'message':'Invalid request value'} for e in exc.errors()]})

    def service(request):
        result=getattr(request.app.state,'replay',None)
        if result is None:
            raise HTTPException(503,'Synthetic demonstration artifacts unavailable; see local setup instructions')
        return result

    @app.get('/health',response_model=HealthResponse)
    def health(request:Request):
        ready=getattr(request.app.state,'replay',None) is not None
        if not ready:
            return JSONResponse(status_code=503,content={'status':'unavailable','synthetic_only':True,'ready':False})
        return {'status':'ok','synthetic_only':True,'ready':True}

    @app.get('/api/metadata',response_model=MetadataResponse)
    def metadata(request:Request):
        return service(request).metadata()

    @app.post('/api/replay',response_model=ReplayResponse)
    def replay(body:ReplayRequest,request:Request):
        engine=service(request)
        try:
            return engine.replay(body)
        except ReplayInputError as exc:
            raise HTTPException(exc.status,str(exc)) from None
        except Exception:
            raise HTTPException(503,'Synthetic inference unavailable') from None

    return app
