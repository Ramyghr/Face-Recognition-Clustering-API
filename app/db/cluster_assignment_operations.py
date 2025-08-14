# db/cluster_assignment_operations.py
import logging
from typing import List, Optional, Dict, Any
from beanie import PydanticObjectId
from bson import ObjectId

from app.models.cluster_assignments import ClusterAssignment

logger = logging.getLogger(__name__)


class ClusterAssignmentDB:
    """Database operations for cluster assignments"""
    
    @staticmethod
    async def create_assignment(
        clustering_id: ObjectId,
        cluster_id: str,
        bucket_path: str,
        user_id: str,
        similarity: float,
        strategy: str = "centroid",
        face_ids: List[str] = None,
        image_paths: List[str] = None
    ) -> ClusterAssignment:
        """Create a new cluster assignment"""
        assignment = ClusterAssignment(
            clustering_id=clustering_id,
            cluster_id=cluster_id,
            bucket_path=bucket_path,
            user_id=user_id,
            similarity=similarity,
            strategy=strategy,
            face_ids=face_ids or [],
            image_paths=image_paths or [],
            face_count=len(face_ids) if face_ids else 0
        )
        await assignment.insert()
        return assignment
    
    @staticmethod
    async def get_assignments_by_clustering(clustering_id: ObjectId) -> List[ClusterAssignment]:
        """Get all assignments for a specific clustering"""
        return await ClusterAssignment.find(
            ClusterAssignment.clustering_id == clustering_id
        ).to_list()
    
    @staticmethod
    async def get_assignments_by_user(user_id: str) -> List[ClusterAssignment]:
        """Get all assignments for a specific user"""
        return await ClusterAssignment.find(
            ClusterAssignment.user_id == user_id
        ).sort("-created_at").to_list()
    
    @staticmethod
    async def get_assignment_by_cluster(
        clustering_id: ObjectId, 
        cluster_id: str
    ) -> Optional[ClusterAssignment]:
        """Get assignment for a specific cluster"""
        return await ClusterAssignment.find_one(
            ClusterAssignment.clustering_id == clustering_id,
            ClusterAssignment.cluster_id == cluster_id
        )
    
    @staticmethod
    async def update_assignment(
        clustering_id: ObjectId,
        cluster_id: str,
        user_id: str,
        similarity: float,
        **kwargs
    ) -> Optional[ClusterAssignment]:
        """Update existing assignment or create new one"""
        assignment = await ClusterAssignment.find_one(
            ClusterAssignment.clustering_id == clustering_id,
            ClusterAssignment.cluster_id == cluster_id
        )
        
        if assignment:
            assignment.user_id = user_id
            assignment.similarity = similarity
            assignment.touch()
            for key, value in kwargs.items():
                setattr(assignment, key, value)
            await assignment.save()
            return assignment
        else:
            return await ClusterAssignmentDB.create_assignment(
                clustering_id=clustering_id,
                cluster_id=cluster_id,
                user_id=user_id,
                similarity=similarity,
                **kwargs
            )
    
    @staticmethod
    async def delete_assignments_by_clustering(clustering_id: ObjectId) -> int:
        """Delete all assignments for a clustering"""
        result = await ClusterAssignment.find(
            ClusterAssignment.clustering_id == clustering_id
        ).delete()
        return result.deleted_count
    
    @staticmethod
    async def get_user_cluster_summary(user_id: str) -> Dict[str, Any]:
        """Get summary of clusters assigned to a user"""
        assignments = await ClusterAssignmentDB.get_assignments_by_user(user_id)
        
        total_clusters = len(assignments)
        total_faces = sum(a.face_count for a in assignments)
        avg_similarity = sum(a.similarity for a in assignments) / total_clusters if total_clusters > 0 else 0
        
        bucket_breakdown = {}
        for assignment in assignments:
            bucket = assignment.bucket_path
            if bucket not in bucket_breakdown:
                bucket_breakdown[bucket] = {"clusters": 0, "faces": 0}
            bucket_breakdown[bucket]["clusters"] += 1
            bucket_breakdown[bucket]["faces"] += assignment.face_count
        
        return {
            "user_id": user_id,
            "total_clusters": total_clusters,
            "total_faces": total_faces,
            "average_similarity": round(avg_similarity, 4),
            "bucket_breakdown": bucket_breakdown,
            "assignments": [
                {
                    "clustering_id": str(a.clustering_id),
                    "cluster_id": a.cluster_id,
                    "bucket_path": a.bucket_path,
                    "similarity": a.similarity,
                    "face_count": a.face_count,
                    "created_at": a.created_at
                }
                for a in assignments
            ]
        }