"""Brain Cancer Type Prediction — FastAPI web service.

Serves an HTML prediction page and a JSON prediction API backed by a trained
Random Forest model. All model artifacts are loaded relative to this file so
the application can be started from any working directory.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# Load the trained model and preprocessing artifacts. They were pickled with
# an older scikit-learn build; see the README for the exact pin needed to
# reproduce this environment without warnings.
MODEL_DIR = os.path.join(BASE_DIR, "models")
_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
_label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


app = FastAPI()

# Enable CORS so the page can be served from any origin locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static images (tumour-type thumbnails) and Jinja2 templates are served from
# the folders next to this file.
app.mount("/images", StaticFiles(directory=os.path.join(BASE_DIR, "images")), name="images")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ---------------------------------------------------------------------------
# Authentication (optional)
# ---------------------------------------------------------------------------
# Key auth is opt-in: set API_KEY in the environment to require a request
# header. Leave it unset to run without authentication (developer-friendly).
API_KEY = os.getenv("API_KEY", "").strip()
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def verify_api_key(api_key: str = Depends(api_key_header)) -> None:
    """Reject requests whose X-API-KEY does not match the configured key."""
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API Key")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class InputData(BaseModel):
    data: list[float]


class PredictionResponse(BaseModel):
    prediction: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: InputData, _auth: None = Depends(verify_api_key)):
    """Return the predicted tumour class for a 3-gene input vector."""
    try:
        scaled_data = _scaler.transform([input_data.data])
        prediction = _model.predict(scaled_data)
        decoded_prediction = _label_encoder.inverse_transform(prediction)
        return PredictionResponse(prediction=decoded_prediction[0])
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again later.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    """Render the prediction page."""
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


@app.get("/home", response_class=HTMLResponse)
async def read_home(request: Request) -> HTMLResponse:
    """Render the homepage."""
    return templates.TemplateResponse(
        request=request, name="home.html", context={"request": request}
    )


@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request) -> HTMLResponse:
    """Render the About/team page."""
    return templates.TemplateResponse(
        request=request, name="about.html", context={"request": request}
    )
