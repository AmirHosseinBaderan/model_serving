# model_serving

A production-ready machine learning model serving system built with Python. This project demonstrates a complete, modular MLOps pipeline that goes from **training** an ML model, through **experiment tracking**, to **exporting** and **serving predictions** over HTTP.

The project is structured in clean, decoupled layers so that each concern — training, tracking, serving, and API — can be developed, tested, and evolved independently.

---

## Table of Contents

- [What Is This Project?](#what-is-this-project)
- [Architecture Overview](#architecture-overview)
- [Layer 1: Tracking](#layer-1-tracking)
- [Layer 2: Model](#layer-2-model)
- [Layer 3: Serving](#layer-3-serving)
- [Layer 4: API](#layer-4-api)
- [Entry Point: run.py](#entry-point-runpy)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Testing](#testing)
- [Docker](#docker)

---

## What Is This Project?

`model_serving` is an end-to-end system that demonstrates how to take a trained model and serve it behind an HTTP API. It solves a concrete problem:

> **How do you go from a trained machine learning model to a reliable, production-grade inference service?**

The project answers this by implementing a full pipeline:

1. **Train** a small neural network on the XOR problem.
2. **Track** the training run (parameters, metrics, artifacts) in MLflow.
3. **Export** the trained PyTorch model into the ONNX format — a portable, framework-agnostic representation.
4. **Serve** the exported model behind a FastAPI web server that exposes `/health`, `/ready`, and `/predict` endpoints.

The model being trained is a **simple 2-input neural network** that learns the XOR logical function — a classic "hello world" for non-linear models. Despite its simplicity, it demonstrates everything needed in a real production serving system: configuration, artifact management, lifecycle management, error handling, and health checks.

---

## Architecture Overview

The codebase is organized into four self-contained layers, plus a small entry point:

```
model_serving/
├── api/                     # Layer 4: HTTP API (FastAPI)
│   ├── app.py               # FastAPI app factory
│   ├── schemas/             # Pydantic request/response models
│   └── routes/              # Health-check & prediction routers
├── serving/                 # Layer 3: Model serving engine
│   ├── bootstrap.py         # Factory — wires the whole stack together
│   ├── config.py          # Environment-based configuration
│   ├── server.py          # High-level ModelServer orchestrator
│   ├── model_service.py   # Service layer (readiness, prediction)
│   ├── inference_engine.py  # Abstract engine (Strategy pattern)
│   ├── onnx_engine.py     # ONNX Runtime implementation
│   ├── pytorch_engine.py  # PyTorch inference implementation
│   ├── predictor.py       # Input transformation & output formatting
│   ├── artifact_resolver.py  # Resolves model files from disk
│   ├── model_identifier.py   # Value object: name + version
│   ├── model_artifact.py     # Value object: resolved artifact
│   ├── model_metadata.py     # Value object: model metadata
│   ├── model_metadata_loader.py  # Loads metadata from JSON
│   └── exceptions.py     # Domain-specific exceptions
├── model/                 # Layer 2: Model definition, training, export
│   ├── model.py           # SimpleModel — the PyTorch neural network
│   ├── train.py           # Training loop
│   ├── export.py          # PyTorch → ONNX export
│   └── artifacts/         # Stored model artifacts (regenerable)
├── tracking/              # Layer 1: Experiment tracking
│   ├── tracker.py         # ExperimentTracker — abstract interface
│   ├── mlflow_tracker.py  # MLflow-backed implementation
│   ├── in_memory_tracker.py  # In-process implementation (testing)
│   ├── experiment.py      # Experiment value object
│   └── run.py             # Run value object
├── run.py                 # Entry point: train → export
├── tests/                 # Full test suite (pytest)
├── Dockerfile             # Production container image
├── requirements.txt       # Full dependency list (dev + runtime)
└── requirements-runtime.txt  # Minimal runtime deps (for Docker)
```

The guiding principle is **separation of concerns via explicit dependency direction**:

```
api  →  serving  →  model
           ↓
       tracking
```

- The **`api`** layer depends on `serving` (it calls `server.predict()`).
- The **`serving`** layer depends on `model` (the PyTorch engine inspects the model class) and produces/reads artifacts in `model/artifacts/`.
- The **`tracking`** layer is a standalone component used by `run.py` and `model/train.py`, but it is never imported by `serving` or `api`. This means inference never depends on your experiment-tracking backend.

---

## Layer 1: Tracking

### What It Does

The `tracking/` package provides an **experiment tracking** abstraction. It records everything you need to reproduce and compare ML training runs:

| Concept        | What Is Tracked                          |
|----------------|------------------------------------------|
| **Experiment** | A named container for related runs       |
| **Run**        | A single training execution              |
| **Parameters** | Hyperparameters (learning rate, epochs)  |
| **Metrics**    | Time-series values (loss per epoch)      |
| **Artifacts**  | Files produced by the run (model weights)|
| **Metadata**   | Arbitrary key-value tags (seed, commit)  |

### How It Works

The layer is built around three ideas:

#### 1. An Abstract Interface (`tracker.py`)

```python
class ExperimentTracker:
    def start_run(self, experiment_name: str) -> Run: ...
    def log_parameter(self, run: Run, name: str, value: Any): ...
    def log_metric(self, run: Run, name: str, value: float, step: int): ...
    def log_artifact(self, run: Run, path: str): ...
    def log_metadata(self, run: Run, name: str, value: Any): ...
    def finish_run(self, run: Run) -> None: ...
    def fail_run(self, run: Run) -> None: ...
    ...
```

`ExperimentTracker` is an **abstract base class** (interface). It defines the contract every tracker implementation must fulfill. This is a textbook **Strategy pattern**: the training code calls `tracker.start_run(...)` without knowing whether the results will be stored in MLflow, a database, or an in-memory dict.

> **Why an interface?** This lets you swap the tracking backend — from a local in-memory store for fast tests to a real MLflow server for production — without changing the training code. The training logic depends on the *abstraction*, not a concrete implementation.

#### 2. Two Concrete Implementations

- **`MLflowExperimentTracker`** (`mlflow_tracker.py`) — persists runs to an MLflow tracking server. Every `log_parameter`, `log_metric`, etc. call delegates to the `mlflow` SDK while also mirroring the data in a local `Run` object so callers can inspect it in-process.
- **`InMemoryExperimentTracker`** (`in_memory_tracker.py`) — stores everything in Python dictionaries. Zero external dependencies, ideal for unit tests and local development.

Both implementations share the same interface, so `model/train.py` can accept either one.

#### 3. Value Objects (`run.py`, `experiment.py`)

```python
@dataclass
class Run:
    id: str
    experiment_name: str
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.RUNNING
    parameters: dict[str, Any] = ...
    metrics: dict[str, list[MetricValue]] = ...
    artifacts: list[str] = ...
    metadata: dict[str, Any] = ...
```

```python
class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
```

A `Run` is a **frozen-ish dataclass** that holds all the state from a training execution. It enforces **lifecycle rules**: the `ensure_active()` method raises `ValueError` if you try to log metrics or artifacts after a run has been finished or failed. This prevents the subtle bug where you accidentally log to a closed run.

> **Why value objects?** Centralizing run state in a dataclass makes it easy to test the tracker logic without any I/O. The tests in `tests/test_experiment_tracker.py` exercise every lifecycle transition using only the `InMemoryExperimentTracker`.

### Design Rationale

| Decision | Why |
|---|---|
| Abstract `ExperimentTracker` interface | Decouples training from any specific tracking backend (MLflow, Weights & Bias, custom DB) |
| `InMemoryExperimentTracker` as a second impl | Makes tests fast and deterministic — no external server required |
| Mirroring MLflow data into the `Run` dataclass | Lets callers read back logged values without an extra SDK round-trip |
| `ensure_active()` lifecycle guard | Prevents logging to closed runs, catching logic errors early |

---

## Layer 2: Model

### What It Does

The `model/` package owns everything related to the **model definition, training, and export**:

- **`model.py`** — Defines `SimpleModel`, the neural network architecture.
- **`train.py`** — The training loop with tracking integration.
- **`export.py`** — Converts the trained PyTorch model into ONNX format.
- **`artifacts/`** — The on-disk output directory for trained weights and exported models.

### The Model: `SimpleModel`

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),   # 2 inputs  → 8 hidden
            nn.ReLU(),
            nn.Linear(8, 1),   # 8 hidden  → 1 output
        )

    def forward(self, x):
        return self.network(x)
```

This is a **2-8-1 feed-forward neural network**. It takes 2 input features, passes them through a hidden layer of 8 neurons with ReLU activation, and produces a single scalar output. It learns the XOR function:

| Input | Input B | Output |
|-------|--------|--------|
| 0     | 0      | 0      |
| 0     | 1      | 1      |
| 1     | 0      | 1      |
| 1     | 1      | 2      |

> **Why XOR?** XOR is the canonical "non-linearly-separable" problem. A single linear layer cannot solve it — you need at least one hidden layer with a non-linear activation. This proves the architecture has enough capacity to learn non-linear mappings.

### How Training Works (`train.py`)

The training function:

1. Sets a fixed random seed (`42`) for reproducibility.
2. Creates the model, optimizer (Adam, lr=0.01), and loss function (MSE).
3. If an `ExperimentTracker` is provided, starts a run and logs hyperparameters as parameters and metadata.
4. Runs the training loop for a configurable number of epochs, logging `train_loss` as a metric after each epoch.
5. Saves the model's `state_dict` to `model/artifacts/<name>/<version>/model.pt`.
6. If tracking is active, logs the saved file as a run artifact and finishes the run. If any error occurs, the run is marked as **failed**.

```python
for epoch in range(epochs):
    prediction = model(x)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    tracker.log_metric(run, "train_loss", loss.item(), step=epoch + 1)
```

> **Why MSE loss?** The XOR targets are {0, 1, 1, 2}. The model regression outputs a single float, so mean squared error directly measures prediction quality as a regression task.
>
> **Why Adam?** Adam adapts learning rates per-parameter automatically, converging faster than plain SGD for small networks like this one.
>
> **Why structured artifact paths (`<name>/<version>/`)?** Versioned directories let you keep multiple model versions side by side. You can deploy `xor:v1` while training `xor:v2`, and the serving layer can switch between them by changing configuration alone.

### How Export Works (`export.py`)

After training, the PyTorch model is exported to ONNX:

```python
torch.onnx.export(
    model,
    dummy_input,           # torch.tensor([[0.0, 0.0]])
    onnx_path,
    input_names=["inputs"],
    output_names=["predictions"],
    dynamic_axes={
        "inputs": {0: "batch_size"},
        "predictions": {0: "batch_size"},
    },
)
```

This produces:
- **`model.onnx`** — The serialized ONNX computation graph.
- **`metadata.json`** — A small JSON file recording the model's name, version, format, and backend.

> **Why ONNX?** ONNX (Open Neural Network Exchange) is an open standard for representing ML models. By exporting to ONNX, the model becomes **framework-agnostic** — it can be served by ONNX Runtime without needing PyTorch installed. This dramatically shrinks the runtime Docker image (see `requirements-runtime.txt`).
>
> **Why `dynamic_axes`?** Marking axis 0 as dynamic (`"batch_size"`) allows the inference engine to accept **any batch size** at runtime — a single input `[0.0, 0.0]` or a thousand inputs. This is essential for a production serving API.
>
> **Why embed metadata?** The serving layer reads `metadata.json` to validate that the artifact on disk matches the requested model name and version before loading it. This prevents accidentally loading the wrong model.

### Artifact Directory Structure

```
model/artifacts/
└── xor/
    ├── v1/
    │   ├── model.onnx       # ONNX graph
    │   ├── model.onnx.data  # External weights (for large models)
    │   └── metadata.json    # { "name": "xor", "version": "v1", "format": "onnx", "backend": "onnxruntime" }
    └── v2/
        ├── model.pt         # PyTorch weights (pre-export)
        ├── model.onnx       # ONNX graph
        ├── model.onnx.data
        └── metadata.json
```

---

## Layer 3: Serving

### What It Does

The `serving/` package is the **inference engine** — it takes a trained model artifact and turns it into a callable predictor. It handles:

- Loading the model file from disk.
- Initializing the inference runtime (ONNX Runtime or PyTorch).
- Accepting input data, running inference, and returning predictions.
- Managing the model lifecycle (start, stop, readiness checks).

### How the Layers Compose

The serving stack is assembled via **dependency injection**, using the **Facade** and **Strategy** patterns:

```
ModelServer          (facade — the public entry point)
  └── ModelService   (service — readiness, error handling)
       └── Predictor (adapter — NumPy ↔ engine I/O conversion)
            └── InferenceEngine   (interface / strategy)
                 ├── ONNXEngine
                 └── PyTorchEngine
```

### The Core Abstraction: `InferenceEngine`

```python
class InferenceEngine(ABC):
    @property
    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray: ...
```

This is the **Strategy** abstraction. Any inference backend — ONNX Runtime, PyTorch, TensorFlow, a remote gRPC model — can be plugged in by implementing this interface.

> **Why an abstract engine?** The API layer, service layer, and predictor all depend on the `InferenceEngine` interface, not on a specific framework. Swapping from ONNX to PyTorch (or adding a new backend) requires **zero changes** to `ModelServer`, `ModelService`, or `Predictor`.

### Concrete Engines

#### `ONNXEngine` (`onnx_engine.py`)

```python
class ONNXEngine(InferenceEngine):
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = None
        self.output_name = None

    def start(self):
        if not self.model_path.exists():
            raise FileNotFoundError(...)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, inputs):
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: inputs},
        )
        return outputs[0]
```

ONNX Engine creates an `onnxruntime.InferenceSession` lazily (on `start()`, not in `__init__`). This **separates object construction from resource loading** — you can instantiate an engine without touching disk, which makes testing trivial.

#### `PyTorchEngine` (`pytorch_engine.py`)

```python
class PyTorchEngine(InferenceEngine):
    def start(self):
        model = SimpleModel()
        state_dict = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def predict(self, inputs):
        tensor = torch.from_numpy(inputs).float()
        with torch.inference_mode():
            predictions = self.model(tensor)
        return predictions.numpy()
```

PyTorch Engine loads the PyTorch model class directly and the saved `state_dict`. It uses `torch.inference_mode()` for efficient, no-grad inference.

> **Why two engines?** ONNX Runtime is lightweight and ideal for production serving (smaller image, cross-platform). PyTorch is useful for **testing** — you can validate that the ONNX export produces the same outputs as the original PyTorch model (see `tests/test_onnx.py::test_onnx_matches_pytorch`).

### `Predictor` — The I/O Adapter (`predictor.py`)

```python
class Predictor:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def predict(self, inputs: list[list[float]]) -> list[float]:
        array = np.asarray(inputs, dtype=np.float32)
        predictions = self.engine.predict(array)
        return predictions.squeeze(-1).tolist()
```

The Predictor is an **adapter** between two worlds:
- **External callers** send `list[list[float]]` (JSON-serializable).
- **Engines** expect `np.ndarray` (NumPy).

The Predictor converts between them, also handling the **shape squeeze**: ONNX/PyTorch output shape is `(batch, 1)`, but callers expect a flat `list[float]`, so `.squeeze(-1)` removes the singleton dimension.

> **Why a separate Predictor?** Keeping the conversion logic in one place means the engine stays pure NumPy in/out, and changing the output format (e.g., returning class labels) only requires editing the Predictor, not every engine implementation.

### `ModelService` — The Service Layer (`model_service.py`)

```python
class ModelService:
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self._predictor = Predictor(engine)

    @property
    def is_ready(self):
        return self.engine.is_ready

    def start(self):
        if self.is_ready:
            return  # idempotent
        self.engine.start()

    def predict(self, inputs):
        if not self.is_ready:
            raise ModelNotReadyError("Model service is not ready.")
        try:
            return self._predictor.predict(inputs)
        except Exception as exc:
            raise InferenceError("Inference failed.") from exc
```

The Service layer adds two crucial pieces of production logic:

1. **Readiness gating** — `predict()` raises `ModelNotReadyError` if the engine hasn't been started.
2. **Error isolation** — Any exception from the engine is caught, wrapped in a domain-specific `InferenceError`, and re-raised. This prevents raw framework exceptions (e.g., ONNXRuntime errors) from leaking to the API layer.

> **Why a separate service layer?** The `ModelServer` (facade) should not need to know about error-wrapping or readiness semantics. The Service layer centralizes cross-cutting concerns that apply regardless of which engine is used.

### `ModelServer` — The Facade (`server.py`)

```python
class ModelServer:
    def __init__(self, engine: InferenceEngine):
        self.service = ModelService(engine)
        self._running = False

    @property
    def is_ready(self):
        return self.service.is_ready

    def start(self):
        if self._running:
            return
        self.service.start()
        self._running = True

    def predict(self, inputs):
        if not self._running:
            raise RuntimeError("Model server is not running.")
        return self.service.predict(inputs)
```

The Server is the **top-level facade**. It adds a `running` flag on top of the service's `ready` flag, distinguishing between "the server process is alive" and "the model is loaded and ready to predict."

> **Why two states (`running` vs `ready`)?** In production, a process might be running (alive) but not yet ready (model still loading, or model file missing). Health-check endpoints (`/health` vs `/ready`) use these two states to communicate different things to orchestrators like Kubernetes: `/health` means "is the process alive" and `/ready` means "can this instance serve traffic."

### Artifact Resolution

The serving layer loads models generically, not with hardcoded paths. Three value objects handle this:

#### `ModelIdentifier` (`model_identifier.py`)

```python
@dataclass(frozen=True)
class ModelIdentifier:
    name: str
    version: str
```

A frozen, validated dataclass. The `__post_init__` method raises `ValueError` if `name` or `version` is empty. Being **frozen** means it's hashable and immutable — safe to use as a dictionary key.

#### `ModelMetadata` (`model_metadata.py`)

```python
@dataclass(frozen=True)
class ModelMetadata:
    name: str
    version: str
    format: str
    backend: str
```

The metadata read from `metadata.json`. Also frozen and validated.

#### `ModelArtifact` (`model_artifact.py`)

```python
@dataclass(frozen=True)
class ModelArtifact:
    identifier: ModelIdentifier
    model_path: Path
    metadata: ModelMetadata
```

Bundles everything the engine needs: *who* (identifier), *where* (model_path), and *what* (metadata).

#### `ModelArtifactResolver` (`artifact_resolver.py`)

```python
class ModelArtifactResolver:
    def __init__(self, artifacts_root: str | Path):
        self.artifacts_root = Path(artifacts_root)
        self.metadata_loader = ModelMetadataLoader()

    def resolve(self, identifier: ModelIdentifier) -> ModelArtifact:
        artifact_directory = self.artifacts_root / identifier.name / identifier.version
        model_path = artifact_directory / "model.onnx"
        # ... validates file exists ...
        metadata = self.metadata_loader.load(artifact_directory / "metadata.json")
        # ... validates metadata matches identifier ...
        return ModelArtifact(identifier, model_path, metadata)
```

The Resolver implements a **directory convention**: `artifacts_root / <name> / <version> / model.onnx` + `metadata.json`. It performs two safety checks:

1. The model file must exist on disk.
2. The embedded metadata's `name` and `version` must match the requested `ModelIdentifier`.

> **Why validate metadata against the identifier?** If someone deploys a mismatched artifact (e.g., copy-pasting the wrong model file), the resolver catches it at startup rather than failing silently with wrong predictions at runtime. This is "fail fast."

### Bootstrap: Wiring It All Together (`bootstrap.py`)

```python
def create_server(config: AppConfig | None = None) -> ModelServer:
    config = config or AppConfig()
    identifier = ModelIdentifier(name=config.model_name, version=config.model_version)

    if config.model_backend == "onnx":
        resolver = ModelArtifactResolver(artifacts_root=config.model_artifacts_root)
        artifact = resolver.resolve(identifier)
        engine = ONNXEngine(artifact.model_path)
    else:
        raise ValueError(f"Unsupported model backend: {config.model_backend}")

    return ModelServer(engine)
```

The `create_server` function is a **factory** that reads configuration, resolves the artifact, instantiates the correct engine, and wires it into a `ModelServer`. The dependency direction is clear: config → identifier → resolver → artifact → engine → service → server.

> **Why a factory instead of inline assembly?** Centralizing the wiring in one place means the API layer (`api/app.py`) just calls `create_server()` and gets a fully-configured server. It also makes the assembly testable — see `tests/test_bootstrap.py`.

### Configuration (`config.py`)

```python
class AppConfig(BaseSettings):
    model_backend: str = "onnx"
    model_name: str
    model_version: str
    model_artifacts_root: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MODEL_",
        case_sensitive=False,
    )
```

Uses **pydantic-settings** to load configuration from environment variables (or a `.env` file). The `case_sensitive=False` setting means `MODEL_NAME`, `model_name`, or `Model_Name` all map to the same setting.

### Domain Exceptions (`exceptions.py`)

```python
class ModelNotReadyError(RuntimeError): ...
class InferenceError(RuntimeError): ...
```

Two custom exceptions with clear semantic meaning:
- `ModelNotReadyError` — raised when a prediction is requested before the model is loaded.
- `InferenceError` — raised when inference itself fails (wrapped from the underlying engine's exception).

> **Why custom exceptions?** They let the API layer register **specific exception handlers** that return structured error responses with HTTP status codes (503 for not-ready, 500 for inference failure), rather than letting raw exceptions produce unhelpful 500 errors.

---

## Layer 4: API

### What It Does

The `api/` package is the **HTTP interface** — a thin FastAPI application that exposes three endpoints and translates HTTP errors into structured JSON responses.

### File Layout

```
api/
├── app.py          # App factory — creates the FastAPI app
├── schemas/
│   ├── prediction.py   # PredictionRequest, PredictionResponse
│   └── error.py        # ErrorResponse, ErrorDetail
└── routes/
    ├── health.py   # /health and /ready endpoints
    └── prediction.py  # /predict endpoint
```

### Request/Response Schemas

```python
# api/schemas/prediction.py
class PredictionRequest(BaseModel):
    inputs: list[list[float]] = Field(min_length=1)

class PredictionResponse(BaseModel):
    predictions: list[float]
```

```python
# api/schemas/error.py
class ErrorResponse(BaseModel):
    error: ErrorDetail  # { code: str, message: str }
```

Pydantic models serve as both **input validation** and **output serialization**. The `Field(min_length=1)` constraint on `inputs` means a request with an empty inputs list is rejected with a `422 Unprocessable Entity` response before any inference runs.

> **Why Pydantic models?** They give you automatic request validation, JSON serialization, and OpenAPI schema generation — FastAPI turns these into a live interactive API docs UI at `/docs`.

### Health Routes (`health.py`)

```python
@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/ready")
def ready():
    if not server.is_ready:
        raise ModelNotReadyError("Model is not ready.")
    return {"ready": True}
```

Two endpoints with distinct semantics:
- **`/health`** — Always returns 200. Confirms the API process is alive. Used by load balancers for liveness checks.
- **`/ready`** — Returns 200 only if the model is loaded. Returns 503 (`ModelNotReadyError`) if the model hasn't finished initializing. Used for readiness checks (e.g., Kubernetes won't route traffic to a pod that isn't ready).

### Prediction Route (`prediction.py`)

```python
@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    predictions = server.predict(request.inputs)
    return PredictionResponse(predictions=predictions)
```

The route receives a validated `PredictionRequest`, delegates to `server.predict()`, and wraps the result in a `PredictionResponse`. The `response_model=PredictionResponse` parameter tells FastAPI to validate and serialize the return value through the Pydantic schema.

### App Factory (`app.py`)

```python
def create_app(server: ModelServer | None = None) -> FastAPI:
    server = server or create_server()

    @asynccontextmanager
    async def lifespan(app):
        server.start()    # initialize the model when the app starts
        yield
        server.stop()     # clean up when the app stops

    app = FastAPI(title="Production AI", version="1.0.0", lifespan=lifespan)

    # Exception handlers — translate domain errors to HTTP responses
    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(request, exc):
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(error={"code": "MODEL_NOT_READY", "message": str(exc)}).model_dump(),
        )

    @app.exception_handler(InferenceError)
    async def inference_error_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error={"code": "INFERENCE_FAILED", "message": str(exc)}).model_dump(),
        )

    app.include_router(create_health_router(server), prefix="/api/v1")
    app.include_router(create_prediction_router(server), prefix="/api/v1")

    return app

app = create_app()  # module-level instance for uvicorn
```

The `create_app` function is a **factory** that:

1. Bootstraps the model server (via `create_server()`).
2. Registers a **lifespan context manager** that starts the server when the application boots and stops it on shutdown — this ensures the model is loaded before the first request and cleaned up gracefully.
3. Registers **exception handlers** that catch domain exceptions (`ModelNotReadyError`, `InferenceError`) and convert them into structured JSON error responses with appropriate HTTP status codes and error codes.
4. Mounts the routers under `/api/v1`.

> **Why a factory?** It accepts an optional `ModelServer` parameter, which makes testing trivial — the test suite injects a `FakeInferenceEngine` so tests don't need real model files. The module-level `app = create_app()` call provides the default instance for `uvicorn`.

### The Request Flow

Here's what happens end-to-end when a client sends a prediction request:

```
Client → POST /api/v1/predict
         ↓
FastAPI validates request body against PredictionRequest (Pydantic)
         ↓
prediction route → server.predict(inputs)
         ↓
ModelServer.predict() → checks _running flag
         ↓
ModelService.predict() → checks is_ready, delegates to Predictor
         ↓
Predictor.predict() → converts list → np.ndarray, calls engine
         ↓
ONNXEngine.predict() → onnxruntime session.run()
         ↓
Predictions flow back: np.ndarray → list[float] → PredictionResponse
         ↓
FastAPI validates response against PredictionResponse, serializes to JSON
         ↓
Client receives { "predictions": [...] }
```

---

## Entry Point: run.py

`run.py` is the orchestration script that ties the **model** and **tracking** layers together:

```python
def main():
    args = parse_args()  # --model-name, --model-version, --experiment-name, --tracking-uri

    tracker = MLflowExperimentTracker(tracking_uri=args.tracking_uri)

    try:
        tracker.get_experiment(args.experiment_name)
    except ValueError:
        tracker.create_experiment(args.experiment_name)

    train(tracker=tracker, experiment_name=..., model_name=..., model_version=...)
    export_model(model_name=..., model_version=...)

# Usage:
# python run.py --model-name xor --model-version v1 --tracking-uri http://127.0.0.1:5000
```

The flow is:
1. Parse command-line arguments.
2. Initialize the MLflow tracker.
3. Ensure the experiment exists (create it if not).
4. **Train** the model (logs parameters, metrics, and artifacts to MLflow).
5. **Export** the trained PyTorch model to ONNX format.
6. The exported ONNX model then becomes available for the serving layer.

> **Why separate train and export?** Training produces PyTorch weights (`model.pt`). Exporting converts them to ONNX. These are deliberately separate steps so that: (a) you can inspect/validate the PyTorch model before committing to ONNX, and (b) the export step is the explicit boundary where the "training framework" dependency is dropped and the "serving format" is committed.

---

## Configuration

The project uses environment variables for configuration, loaded via a `.env` file:

```env
MODEL_BACKEND=onnx
MODEL_NAME=xor
MODEL_VERSION=v1
MODEL_ARTIFACTS_ROOT=./model/artifacts
```

These map to the `AppConfig` class in `serving/config.py`. The variable naming convention is `MODEL_` prefix + lowercase setting name.

| Variable | Description | Default |
|---|---|---|
| `MODEL_BACKEND` | Inference backend: `onnx` or `pytorch` | `onnx` |
| `MODEL_NAME` | Model name (used for artifact path) | *(required)* |
| `MODEL_VERSION` | Model version (used for artifact path) | *(required)* |
| `MODEL_ARTIFACTS_ROOT` | Root directory for model artifacts | *(required)* |

---

## How to Run

### Prerequisites

- Python 3.11+
- An MLflow tracking server running at `http://127.0.0.1:5000` (for the full pipeline with tracking)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MLflow Tracking Server (optional)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### 3. Train and Export a Model

```bash
python run.py \
  --model-name xor \
  --model-version v1 \
  --experiment-name xor \
  --tracking-uri http://127.0.0.1:5000
```

This trains the XOR model, logs everything to MLflow, and exports an ONNX file to `model/artifacts/xor/v1/model.onnx`.

### 4. Start the API Server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 5. Make Predictions

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[0, 0], [0, 1], [1, 0], [1, 1]]}'
```

Response:
```json
{
  "predictions": [0.0, 1.0, 1.0, 2.0]
}
```

---

## Testing

The project includes a comprehensive pytest suite covering all layers.

### Run All Tests

```bash
.venv/bin/pytest tests/ -v
```

### Test Organization

| Test File | Layer | What It Covers |
|---|---|---|
| `test_experiment_tracker.py` | Tracking | InMemory tracker lifecycle, validation, metric history |
| `test_mlflow_tracker.py` | Tracking | MLflow persistence (parameters, metrics, tags, artifacts) |
| `test_model_serving.py` | Serving | ModelService lifecycle, readiness, error wrapping |
| `test_server.py` | Serving | ModelServer start/stop/idempotency |
| `test_bootstrap.py` | Serving | Server factory, backend selection, artifact resolution |
| `test_onnx_engine.py` | Serving | ONNX engine load and predict |
| `test_onnx.py` | Model | ONNX model loads, predicts, and matches PyTorch output |
| `test_predictor.py` | Serving | Predictor input/output conversion |
| `test_artifact_resolver.py` | Serving | Artifact path resolution, metadata validation |
| `test_model_identifier.py` | Serving | ModelIdentifier validation |
| `test_model_metadata.py` | Serving | ModelMetadata validation |
| `test_model_metadata_loader.py` | Serving | JSON metadata file loading |
| `test_model_artifact.py` | Serving | ModelArtifact dataclass |
| `test_api.py` | API | Health, readiness, predict, error handling, invalid input |
| `test_training_tracking.py` | Model | End-to-end training with InMemory tracking |
| `test_run.py` | Entry point | Pipeline orchestration (train → export) |

### Testing Strategy

The test suite uses two complementary approaches:

1. **Fake objects** — `tests/fakes.py` provides `FakeInferenceEngine` and `FailingInferenceEngine` that implement the `InferenceEngine` interface without any real model files. This lets `test_api.py` exercise the full HTTP stack (including exception handlers and status codes) without loading an actual model.

2. **Integration tests** — Tests like `test_onnx.py` and `test_model_serving.py` load real model artifacts from `model/artifacts/` and verify end-to-end behavior, including numerical equivalence between PyTorch and ONNX outputs.

---

## Docker

A production-ready Dockerfile is provided:

```Dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser
COPY api ./api
COPY model ./model
COPY serving ./serving
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ready', timeout=3)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key design decisions:
- **`requirements-runtime.txt`** is used instead of the full `requirements.txt`. Only FastAPI, Uvicorn, ONNX Runtime, NumPy, and Pydantic are installed — **PyTorch and MLflow are not needed at inference time** since the model is already exported to ONNX.
- **Non-root user** (`appuser`) — follows security best practices.
- **Health check** against `/api/v1/ready` — ensures the model is loaded before the container is considered healthy by the orchestrator.

Build and run:

```bash
docker build -t model_serving .
docker run -p 8000:8000 model_serving
```
