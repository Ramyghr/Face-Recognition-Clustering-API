from .face_clustering_models import (
    FaceEmbeddingBase as FaceEmbedding,
    ClusterInfo,
    ClusteringResult
)

# Lazy loader for operations to avoid circular imports
_operations_loaded = False
_operations = {}

def _load_operations():
    global _operations_loaded, _operations
    if not _operations_loaded:
        from app.db.face_clustering_operations import (
            get_face_embedding_model,
            get_clustering_result_model,
            DatabaseSynchronizer,
            #UserProfileDB,
            FaceClusteringDB,
            ConnectionManager,
            init_face_clustering_db
        )
        from app.db.user_profile_operations import UserProfileDB
        _operations.update({
            'get_face_embedding_model': get_face_embedding_model,
            'get_clustering_result_model': get_clustering_result_model,
            'DatabaseSynchronizer': DatabaseSynchronizer,
            'FaceClusteringDB': FaceClusteringDB,
            'UserProfileDB': UserProfileDB,
            'ConnectionManager': ConnectionManager,
            'init_face_clustering_db': init_face_clustering_db
        })
        _operations_loaded = True

# Wrapper functions for callables
def get_face_embedding_model(*args, **kwargs):
    _load_operations()
    return _operations['get_face_embedding_model'](*args, **kwargs)

def get_clustering_result_model(*args, **kwargs):
    _load_operations()
    return _operations['get_clustering_result_model'](*args, **kwargs)

def init_face_clustering_db(*args, **kwargs):
    _load_operations()
    return _operations['init_face_clustering_db'](*args, **kwargs)

# Classes can be accessed directly
def __getattr__(name):
    if name in ['DatabaseSynchronizer', 'FaceClusteringDB', 
               'UserProfileDB', 'ConnectionManager']:
        _load_operations()
        return _operations[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")

# Re-export all important classes and functions
__all__ = [
    'FaceEmbedding',
    'ClusterInfo',
    'ClusteringResult',
    'UserProfile',
    'get_face_embedding_model',
    'get_clustering_result_model',
    'DatabaseSynchronizer',
    'FaceClusteringDB',
    'UserProfileDB',
    'ConnectionManager',
    'init_face_clustering_db'
]