import uuid
import logging
from typing import Dict, List, Any, Optional
import datetime
import math # Added for scan distance calculation

# Setup logging
from ..utils.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class MockWeaviateClient:
    """
    Mock implementation of WeaviateClient for testing without a real Weaviate instance.
    """
    
    def __init__(self, url: str, api_key: Optional[str] = None):
        """
        Initialize the mock Weaviate client.
        
        Args:
            url: URL (ignored in mock)
            api_key: API key (ignored in mock)
        """
        self.collections = {}
        logger.info("Initialized MockWeaviateClient")

    def set_llm_manager(self, llm_manager) -> None:
        """Mock implementation of set_llm_manager."""
        pass
    
    def create_collection_if_not_exists(self, collection_name: str) -> None:
        """
        Create a collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection to create
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = {}
            logger.info(f"Created mock collection: {collection_name}")

    def add_object(self, collection_name: str, properties: Dict[str, Any], uuid: Optional[str] = None, vector: Optional[List[float]] = None) -> str:
        """
        Add an object to a collection.
        
        Args:
            collection_name: Name of the collection
            properties: Properties of the object
            uuid: Optional UUID for the object.
            vector: Optional pre-computed vector (ignored in mock).

        Returns:
            ID of the created object (as string)
        """
        # Create collection if it doesn't exist
        self.create_collection_if_not_exists(collection_name)

        # Generate UUID if not provided
        obj_uuid_str = str(uuid) if uuid else str(uuid.uuid4())
        properties['id'] = obj_uuid_str # Ensure ID is in properties for consistency

        # Store the object
        self.collections[collection_name][obj_uuid_str] = properties
        logger.info(f"Added object to mock collection {collection_name} with ID {obj_uuid_str}")

        return obj_uuid_str
    def get_object(self, collection_name: str, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an object by ID.
        
        Args:
            collection_name: Name of the collection
            object_id: ID of the object (string or UUID)

        Returns:
            The object or None if not found
        """
        obj_id_str = str(object_id)
        if collection_name not in self.collections or obj_id_str not in self.collections[collection_name]:
            logger.warning(f"Object {obj_id_str} not found in mock collection {collection_name}")
            return None

        return self.collections[collection_name][obj_id_str]
    def delete_object(self, collection_name: str, object_id: str) -> bool:
        """
        Delete an object by ID.
        
        Args:
            collection_name: Name of the collection
            object_id: ID of the object (string or UUID)

        Returns:
            True if successful, False otherwise
        """
        obj_id_str = str(object_id)
        if collection_name not in self.collections or obj_id_str not in self.collections[collection_name]:
            logger.warning(f"Attempted to delete non-existent object {obj_id_str} from mock collection {collection_name}")
            return False

        del self.collections[collection_name][obj_id_str]
        logger.info(f"Deleted object {obj_id_str} from mock collection {collection_name}")
        return True
    def _filter_objects(self, objects: Dict[str, Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter objects based on filter criteria.
        
        Args:
            objects: Dictionary of objects
            filters: Filter criteria
            
        Returns:
            List of objects that match the filter
        """
        if not filters:
            return list(objects.values())
        
        filtered_objects = []
        
        for obj in objects.values():
            matches = True
            
            # Check category filter
            if "category" in filters and obj.get("category") != filters["category"]:
                matches = False
            
            # Check min importance filter
            if "min_importance" in filters and obj.get("importance", 0) < filters["min_importance"]:
                matches = False
            
            # Check max importance filter
            if "max_importance" in filters and obj.get("importance", 10) > filters["max_importance"]:
                matches = False
            
            if matches:
                filtered_objects.append(obj)
        
        return filtered_objects
    
    def semantic_search(self, collection_name: str, query: str, 
                        filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a mock semantic search.
        
        Args:
            collection_name: Name of the collection
            query: Query text
            filters: Optional filters
            limit: Maximum number of results
            
        Returns:
            List of matching objects
        """
        if collection_name not in self.collections:
            return []
        
        # In a real implementation, this would use vector similarity
        # For the mock, we'll just do a simple text search on the summary field
        results = []
        
        for obj in self.collections[collection_name].values():
            summary = obj.get("summary", "")
            if query.lower() in summary.lower():
                results.append(obj)

        # Apply filters
        results = self._filter_objects({obj["id"]: obj for obj in results}, filters)

        # Sort by recency (newest first) as a proxy for relevance
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return results[:limit]
    
    def keyword_search(self, collection_name: str, query: str, 
                       filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a mock keyword search.
        
        Args:
            collection_name: Name of the collection
            query: Keyword query
            filters: Optional filters
            limit: Maximum number of results
            
        Returns:
            List of matching objects
        """
        if collection_name not in self.collections:
            return []
        
        # Split query into keywords
        keywords = [k.strip().lower() for k in query.split()]
        
        results = []
        for obj in self.collections[collection_name].values():
            # Check if any keyword matches in summary or keywords
            summary = obj.get("summary", "").lower()
            obj_keywords = [k.lower() for k in obj.get("keywords", [])]
            
            for keyword in keywords:
                if keyword in summary or any(keyword in k for k in obj_keywords):
                    results.append(obj)
                    break

        # Apply filters
        results = self._filter_objects({obj["id"]: obj for obj in results}, filters)

        # Sort by importance (highest first)
        results.sort(key=lambda x: x.get("importance", 0), reverse=True)
        
        return results[:limit]
    
    def hybrid_search(self, collection_name: str, query: str, keyword_query: str,
                      filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a mock hybrid search.
        
        Args:
            collection_name: Name of the collection
            query: Semantic query
            keyword_query: Keyword query
            filters: Optional filters
            limit: Maximum number of results
            
        Returns:
            List of matching objects
        """
        # Combine results from semantic and keyword search
        semantic_results = self.semantic_search(collection_name, query, filters, limit=limit*2)
        keyword_results = self.keyword_search(collection_name, keyword_query, filters, limit=limit*2)

        # Combine and deduplicate
        combined_results = {}
        for obj in semantic_results + keyword_results:
            if obj["id"] not in combined_results:
                combined_results[obj["id"]] = obj

        # Sort by a combination of recency and importance
        results = list(combined_results.values())
        
        # Custom sorting function that considers both importance and recency
        def sort_key(obj):
            importance = obj.get("importance", 5)
            
            # Parse timestamp if available
            timestamp = obj.get("timestamp", "")
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                # Convert to a numeric value (seconds since epoch)
                timestamp_value = dt.timestamp()
            except (ValueError, TypeError):
                timestamp_value = 0
            
            # Combine importance and recency (with importance having more weight)
            return (importance * 10000) + timestamp_value
        
        results.sort(key=sort_key, reverse=True)
        
        return results[:limit]
