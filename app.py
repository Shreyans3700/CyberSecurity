import os
import sys
import certifi
import pandas as pd
from dotenv import load_dotenv
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.pipeline.training_pipeline import TrainingPipeline
from network_security.constants.model_trainer import ModelTrainerArtifact
from network_security.utils.utils import load_object
from network_security.utils.ml_utils import NetworkModel

from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, UploadFile, File
from uvicorn import run as app_run
from starlette.responses import RedirectResponse

ca = certifi.where()
templates = Jinja2Templates(directory="./templates")

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


@app.post("/predict")
async def predict_pipeline(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_models/preprocessor.pkl")
        model = load_object("final_models/model.pkl")

        network_model = NetworkModel(preprocessor=preprocessor, model=model)

        y_pred = network_model.predict(df)

        print(y_pred)

        df["predicted_column"] = y_pred

        table_html = df.to_html(classes="table table-striped")

        return templates.TemplateResponse(
            request=request,
            name="template.html",
            context={"table": table_html}
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
