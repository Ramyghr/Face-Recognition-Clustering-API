# 🧠 Face Recognition & Clustering API

*Poetry + FastAPI + MongoDB + Qdrant. Pragmatic, fast, production-minded.*

> You give it a bucket of images. It finds faces, builds **person clusters** (every appearance of the same person across images), and **assigns** those clusters to **known users**.
> **Vectors** live in **Qdrant**, **metadata** in **MongoDB**. Endpoints are clean and debug-friendly.

---

## 📌 Overview

A **FastAPI-based facial recognition and clustering system** integrating:

* **YOLO** for face detection (onnx/pt supported)
* **FaceNet** for embedding generation (ONNX)
* **MongoDB** for metadata storage
* **Qdrant** for vector similarity search
* **Wasabi S3** (optional) for cloud image storage

### Core Capabilities

* Create/update user profiles with validated face embeddings
* Cluster large batches of images from cloud storage
* Assign anonymous person clusters to known profiles
* Debug tools for development
* Quality control on image inputs

---

## ✨ Features

### 1) User Profile Management

* Detect and validate a single high-quality face from an image
* Generate and store embeddings in MongoDB
* Adjustable quality parameters: `min_face_size`, `min_confidence`, `blur_threshold`
* Debug endpoints for development

### 2) Person-Based Clustering

* Batch process Wasabi S3 (or local) images
* Detect **all** faces per image & extract embeddings
* Graph-based clustering with cosine similarity
* Store:

  * **Metadata → MongoDB**
  * **Embeddings → Qdrant**
* Verify storage consistency

### 3) Cluster → User Assignment

* Match clusters to user profiles based on cosine similarity
* Support for `centroid` (default) and `vote` strategies
* CRUD & summaries for assignments
* Debug endpoints to inspect collections and documents

---

## 🏗 Architecture

```
[Images in Wasabi/local] ---- download ----+
                                           |
                                   [YOLO face detector]
                                           |
                                        [Crops]
                                           |
                                [FaceNet embedding model]
                                           |
                      +--------------------+--------------------+
                      |                                         |
               [Qdrant Vector DB]                       [MongoDB Metadata]
               embeddings + payloads:                   - user_profiles
               face_id, image_path, person_id,         - person_faces_{bucket}_{sub}
               bucket/sub, bbox, conf, quality         - person_clustering_{bucket}_{sub}
                                                        - cluster_assignments
```

---

## 🛠 Tech Stack

| Component      | Technology                                                     |
| -------------- | -------------------------------------------------------------- |
| API Framework  | FastAPI                                                        |
| Face Detection | YOLO (yolov11n-face.onnx preferred, yolov8n-face.pt supported) |
| Embeddings     | FaceNet (ONNX)                                                 |
| Metadata DB    | MongoDB (Beanie + Motor fallback)                              |
| Vector DB      | Qdrant                                                         |
| Storage        | Wasabi S3 (optional)                                           |
| Processing     | OpenCV, Pillow, pillow-heif                                    |
| Clustering     | Graph + cosine (legacy DBSCAN path exists)                     |
| Concurrency    | asyncio, ThreadPoolExecutor                                    |

---

## 📦 Requirements

* Python **3.10+** (Poetry-managed)
* **MongoDB** (local/remote)
* **Qdrant** (Docker)
* Model files in `./app/ai_models/`:

  * `Facenet.onnx`
  * `vgg-tf2onnx.onnx` (optional)
  * `yolov11n-face.onnx` (preferred) **or** `yolov8n-face.pt`

---

## ⚙️ Installation & Setup

### Prerequisites

* Python **3.10+**
* Docker
* Poetry
* MongoDB (local/cloud)
* Qdrant (local via Docker or cloud)

### 1) Clone the repo

```bash
git clone https://github.com/Ramyghr/Face-Recognition-Clustering-API.git
cd Face-Recognition-Clustering-API
```

### 2) Install dependencies (Poetry)

```bash
poetry install
```

### 3) Models

Place your models in `app/ai_models/`:

```
app/ai_models/
├─ Facenet.onnx
├─ vgg-tf2onnx.onnx         # optional
├─ yolov11n-face.onnx       # preferred
└─ yolov8n-face.pt          # alternative
```

### 4) Environment — create `.env.dev`

> This repo reads `.env.dev` when `ENV=dev`.

```dotenv
ENV=dev
DEEPFACE_HOME=./app/ai_models

# Local storage (optional helpers)
LOCAL_STORAGE_PATH=./test-data
CLUSTER_INPUT_PATH=./local-data/input
CLUSTER_OUTPUT_PATH=./local-data/clustered

# Models
EMBED_MODEL_PATH=./app/ai_models/Facenet.onnx
EMBED_MODEL_NAME=FACENET
YOLO_MODEL_PATH=./app/ai_models/yolov11n-face.onnx
# Or override with PT:
# YOLO_MODEL_PATH=./app/ai_models/yolov8n-face.pt

# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGO_URI=mongodb://localhost:27017
MONGO_DB=uwas-recette
MONGO_DB_NAME=uwas-recette

# CORS (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:5001

# Wasabi S3 (optional)
WASABI_ENDPOINT=
WASABI_ACCESS_KEY=
WASABI_SECRET_KEY=
WASABI_BUCKET_NAME=
WASABI_REGION=

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 5) Run Qdrant (Docker)

```bash
docker run --rm -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

### 6) (Optional) Run MongoDB locally (Docker)

```bash
docker run -d --name mongo -p 27017:27017 mongo:7
```

### 7) Launch the API (Poetry + Uvicorn)

```bash
poetry run uvicorn app.main:app --reload
```

Open docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 📡 API Overview

### 🧍 User Profile Endpoints

| Method | Endpoint                   | Description                                                    |
| ------ | -------------------------- | -------------------------------------------------------------- |
| POST   | `/detect-face`             | Detect a face, validate quality, and create a new user profile |
| PUT    | `/update-face/{user_id}`   | Update an existing profile with a new face image               |
| GET    | `/debug-quality-settings`  | View current detection quality thresholds                      |
| GET    | `/debug-profiles`          | List stored profiles (sanitized)                               |
| GET    | `/debug-profile/{user_id}` | Retrieve profile details (sanitized)                           |
| DELETE | `/debug-profile/{user_id}` | Delete a profile                                               |

---

### 🧑‍🤝‍🧑 Person-Based Clustering Endpoints

| Method | Endpoint                                                 | Description                                                 |
| ------ | -------------------------------------------------------- | ----------------------------------------------------------- |
| POST   | `/cluster-bucket-persons/{bucket_name}/{sub_bucket}`     | Run person-based clustering on bucket images                |
| GET    | `/person-cluster-status/{bucket_name}/{sub_bucket}`      | View clustering status/history (supports `clustering_id`)   |
| GET    | `/person-details/{bucket_name}/{sub_bucket}/{person_id}` | Detailed info for one person (owner face, appearances)      |
| DELETE | `/cleanup-person-data/{bucket_name}/{sub_bucket}`        | Cleanup a specific clustering (assignments/Qdrant optional) |

---

### 🔗 Cluster → User Assignment Endpoints

| Method | Endpoint                                                    | Description                                                 |
| ------ | ----------------------------------------------------------- | ----------------------------------------------------------- |
| POST   | `/assignments/{bucket_name}/{sub_bucket}?clustering_id=...` | Assign clusters to known users (cosine vs. owner embedding) |
| POST   | `/assignments/batch`                                        | Assign all clustering docs in a collection                  |
| GET    | `/assignments/{bucket_name}/{sub_bucket}?clustering_id=...` | List assignments for a specific run                         |
| GET    | `/assignments/user/{user_id}`                               | All assignments for a user (grouped by run)                 |
| DELETE | `/assignments/clustering/{clustering_id}`                   | Delete all assignments for a run                            |
| GET    | `/assignments/debug/collections`                            | Inspect clustering collections available                    |
| GET    | `/assignments/debug/collection/{collection_name}`           | Inspect a single collection’s structure                     |
| GET    | `/assignments/debug/test-db-connection`                     | Test DB connectivity                                        |

---

## 📂 Data & Collections

* **Users**: `user_profiles`
* **Faces (person mode)**: `person_faces_{bucket}_{sub}`
* **Clustering (person mode)**: `person_clustering_{bucket}_{sub}`
* **Assignments**: `cluster_assignments`

**Qdrant payload (person mode)**

```json
{
  "face_id": "<uuid>",
  "image_path": "<bucket>/<key>",
  "person_id": "person_# | unassigned",
  "is_owner_face": "true|false",
  "clustering_id": "<mongodb ObjectId string>",
  "bucket_name": "<bucket>",
  "sub_bucket": "<sub>",
  "bbox": "[x,y,w,h]",
  "confidence": "<float as str>",
  "quality_score": "<float as str>",
  "cluster_confidence": "<float as str>"
}
```

---

## 🔬 Pipeline Details

* **Clustering**: detect → embed → graph edges at cosine **≥ 0.75** → merge person clusters by centroid sim **≥ 0.85** → choose **owner face** by `(0.6 * quality + 0.4 * confidence)` → persist to Mongo/Qdrant
* **Assignment**: load user profile embeddings → cosine vs. cluster `owner_embedding` → assign if `≥ threshold` (typ. 0.65–0.7)

---

## ⚡ Performance & Tuning

* `batch_size` ≤ **50**, `max_concurrent` ≤ **12**
* Keep quality filtering **on** for cleaner embeddings
* Big user base? Consider prefiltering candidates via Qdrant ANN before cosine

---

## 🧪 Quick Smoke Tests (cURL)

**Create/Update a profile**

```bash
curl -X POST "http://localhost:8000/detect-face" \
  -F "user_id=test.user@demo" \
  -F "file=@/path/to/face.jpg" \
  -F "min_face_size=30" -F "min_confidence=0.4" -F "blur_threshold=50"
```

**Run person clustering**

```bash
curl -X POST "http://localhost:8000/cluster-bucket-persons/my-bucket/my-sub"
```

**Assign clusters to users**

```bash
curl -X POST "http://localhost:8000/assignments/my-bucket/my-sub?clustering_id=<ObjectId>&threshold=0.7&strategy=centroid"
```

---

## 🆘 Troubleshooting

* **Qdrant not reachable** → ensure Docker command above is running and `QDRANT_HOST/PORT` match `.env.dev`
* **Mongo timeouts** → check `MONGODB_URL` and port `27017` (container/remote)
* **.env not loading** → ensure `ENV=dev` and file is named **.env.dev** at repo root
* **YOLO path mismatch** → choose **one** model path (ONNX or PT) in `.env.dev`

---

## 💻 Dev Tips

* Hot reload:

```bash
poetry run uvicorn app.main:app --reload
```

* Handy checklist:

  * Models in `app/ai_models/`
  * `.env.dev` created (above template)
  * Qdrant Docker running
  * MongoDB running
  * Open **[http://localhost:8000/docs](http://localhost:8000/docs)**

---
