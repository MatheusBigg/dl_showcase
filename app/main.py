from fastapi import FastAPI
from loguru import logger

from app.config.settings import settings

app = FastAPI(
    title='Deep Learning SHOWCASE',
    version=settings.VERSION,
    description='API for showcasing my Deep Learning Projetcs!',
)


@app.on_event('startup')
async def startup():
    logger.info(f'Starting server in {settings.ENVIRONMENT} mode')


@app.get('/')
async def health_check():
    return {
        'status': 'running',
        'version': settings.VERSION,
        'environment': settings.ENVIRONMENT,
        'message': 'Welcome to Deep Learning SHOWCASE!',
    }


@app.get('/welcome')
async def root():
    return {'message': 'Welcome to Deep Learning SHOWCASE!'}
