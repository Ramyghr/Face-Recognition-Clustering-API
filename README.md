# 🧠 Face Recognition & Clustering API

## 📌 Overview

A **FastAPI-based facial recognition and clustering system** integrating:

- **YOLOv8** for face detection
- **FaceNet** for embedding generation
- **MongoDB** for metadata storage
- **Qdrant** for vector similarity search
- **Wasabi S3** for cloud image storage

### Core Capabilities
- Create/update user profiles with validated face embeddings
- Cluster large batches of images from cloud storage
- Assign anonymous face clusters to known profiles
- Debug tools for development
- Quality control on image inputs

---

## ✨ Features

### 1. User Profile Management
- Detect and validate a single high-quality face from an image
- Generate and store embeddings in MongoDB
- Adjustable quality parameters: `min_face_size`, `min_confidence`, `blur_threshold`
- Debug endpoints for development

### 2. Bucket Clustering
- Batch process Wasabi S3 images
- Detect faces & extract embeddings
- Cluster faces with DBSCAN
- Store:
  - Metadata → MongoDB
  - Embeddings → Qdrant
- Verify storage consistency

### 3. Cluster Assignment
- Match clusters to user profiles based on similarity
- Support for `centroid` and `vote` matching strategies
- Full CRUD for assignments
- Statistics and summaries per user

---

## 🛠 Tech Stack

| Component      | Technology |
|----------------|------------|
| API Framework  | FastAPI    |
| Face Detection | YOLOv8 (ONNX) |
| Embeddings     | FaceNet (ONNX) |
| Metadata DB    | MongoDB (Beanie ODM) |
| Vector DB      | Qdrant     |
| Storage        | Wasabi S3  |
| Processing     | OpenCV, Pillow, pillow-heif |
| Clustering     | scikit-learn DBSCAN |
| Concurrency    | asyncio, ThreadPoolExecutor |

---


## ⚙️ Installation & Setup

### Prerequisites
- Python **3.9+**
- Docker
- Poetry
- MongoDB (local/cloud)
- Qdrant (local via Docker or cloud)

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Install Dependencies
poetry install
3. Configure Environment
Create .env or .env.dev:
# MongoDB
MONGODB_URL="mongodb://localhost:27017"
DATABASE_NAME="face_recognition"
# Qdrant
QDRANT_URL="http://localhost:6333"
# Wasabi S3
WASABI_ACCESS_KEY="your-access-key"
WASABI_SECRET_KEY="your-secret-key"
4. Start Qdrant with Docker
docker run --rm -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
5. Run the API
poetry run uvicorn app.main:app --reload
📡 API Overview
User Profile Endpoints
Method	Endpoint	Description
POST	/detect-face	Detect & create user profile
PUT	/update-face/{user_id}	Update existing profile
GET	/debug-quality-settings	View detection thresholds
GET	/debug-profiles	List stored profiles

Clustering Endpoints
Method	Endpoint	Description
POST	/cluster-bucket-with-db/{bucket}/{sub}	Run clustering
GET	/verify-fixed-storage/{bucket}/{sub}	Verify Mongo/Qdrant sync
GET	/cluster-status-db/{bucket}/{sub}	View clustering status
GET	/cluster-details-db/{bucket}/{sub}/{cluster_id}	Cluster details

Cluster Assignment Endpoints
Method	Endpoint	Description
POST	/assignments/{bucket}/{sub}	Assign clusters to users
GET	/assignments/{bucket}/{sub}	View assignments
GET	/assignments/user/{user_id}	View user’s assignments
DELETE	/assignments/clustering/{clustering_id}	Delete assignments

