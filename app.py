import os
import sys
import certifi
from dotenv import load_dotenv
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.pipeline.training_pipeline import TrainingPipeline
from network_security.constants.model_trainer import ModelTrainerArtifact

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from uvicorn import run as app_run
from starlette.responses import RedirectResponse

ca = certifi.where()


load_dotenv()

modeldb_url = os.getenv("MONGODB_URL_KEY")

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_pipeline():
    try:
        train_pipeline = TrainingPipeline()
        train_artifact, model = train_pipeline.run_pipeline()
        logging.info(train_artifact.model_file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
