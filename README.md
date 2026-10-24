# Brain Cancer Type Prediction

A small web application that predicts the type of brain tumour from the
expression values of three genes, using a trained **Random Forest** machine
learning model.

## How it works

The user enters the expression values of three genes and the server returns
the predicted tumour class. The model is a scikit-learn `RandomForestClassifier`
with a fitted `StandardScaler` (3 features) and a `LabelEncoder` encoding five
possible classes.

## Features

- 3-gene input (`NACC2`, `EFCAB1`, `CAMK2N2`)
- FastAPI backend + Random Forest model
- 5 prediction classes:
  - `ependymoma` (index 0)
  - `glioblastoma` (index 1)
  - `medulloblastoma` (index 2)
  - `normal` (index 3)
  - `pilocytic_astrocytoma` (index 4)
- Web pages: `/` (prediction), `/home` (homepage), `/about` (team)
- Static image serving for tumour-type thumbnails

> **Medical disclaimer:** This is a prototype built for education and
> demonstration. It is **not** a medically validated diagnostic tool. Do not
> use it to make clinical decisions.

## Run locally

1. Install Python 3.9+ (the versions in `requirements.txt` are tested and
   verified).
2. Open a terminal in this project folder and install the dependencies:

   ```cmd
   pip install -r requirements.txt
   ```

3. (Optional) Create a `.env` file to enable request-level authentication.
   `.env.example` is provided as a template. **Keep `.env` out of Git** (it is
   already git-ignored) and **never commit real secrets**.
4. Start the server:

   ```cmd
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

5. Open <http://127.0.0.1:8000/> in your browser.

The application runs without `.env` too; authentication is opt-in. See the
Security section.

## API

| Method | Path       | Description                     |
| ------ | ---------- | ------------------------------- |
| `POST` | `/predict` | Return a tumour prediction      |
| `GET`  | `/`        | Render the prediction page      |
| `GET`  | `/home`    | Render the homepage             |
| `GET`  | `/about`   | Render the About / team page    |

### `/predict` request

**Body** (`application/json`):

```json
{
  "data": [gene1_value, gene2_value, gene3_value]
}
```

**Response** (`application/json`):

```json
{
  "prediction": "glioblastoma"
}
```

> Authentication is optional. If you set `API_KEY` in `.env`, clients must send
> the matching key in the `X-API-KEY` header; otherwise the endpoint runs
> without authentication. The browser page no longer ships a hard-coded key.

## Deployment

The `Procfile` is for hosting platforms such as Heroku, Railway, or Render:

```text
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

On Render, set the port with an environment variable named `$PORT`.

## Project structure

```
├── main.py                  # FastAPI application (entry point)
├── requirements.txt         # Python dependencies (pinned)
├── Procfile                 # Deployment configuration (Heroku/Railway/Render)
├── .env.example             # Template for environment variables
├── .gitignore               # Git ignore rules
├── README.md
├── templates/
│   ├── index.html           # Prediction page (root "/")
│   ├── home.html            # Homepage ("/home")
│   └── about.html           # About / team page ("/about")
├── images/                  # Tumour-type thumbnails and charts
└── models/                  # Serialised model artifacts
    ├── random_forest_model.pkl
    ├── scaler.pkl
    └── label_encoder.pkl
```

- `__pycache__/` is git-ignored (generated automatically).
- `models/*.pkl` are committed intentionally (tracked in Git). If you want to
  stop tracking them, uncomment the `*.pkl` line in `.gitignore`.
- `.env` and `.env.local` are git-ignored (secrets).

## Security notes

- `.env` holds `API_KEY` and **must never** be committed. `.env.example` is the
  template for developers.
- No API key is embedded in the browser JavaScript. The page sends only the
  prediction payload to `/predict`; the server applies authentication when it
  is enabled.
- Run the app over HTTPS in production so traffic to `/predict` is encrypted.

## Technologies

- Python 3.9+
- FastAPI, Uvicorn, Starlette, Jinja2, pydantic (v2)
- scikit-learn, joblib, numpy, scipy
- HTML5, CSS, vanilla JavaScript
