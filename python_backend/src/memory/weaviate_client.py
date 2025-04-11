import uuid
import logging
from typing import Dict, List, Any, Optional
import weaviate
import weaviate.classes as wvc # Import classes for v4 syntax

# Setup logging
from ..utils.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from ..llm.llm_manager import LLMManager

class WeaviateClient:
    """
    Client for interacting with Weaviate vector database.
    """
    
    def __init__(self, url: str, api_key: Optional[str] = None):
        """
        Initialize the Weaviate client.
        
        Args:
            url: URL of the Weaviate instance
            api_key: Optional API key for authentication
        """
        self.url = url
        self.api_key = api_key
        self.llm_manager = None  # Will be set later if needed for embeddings
        
        # Configure auth
        auth_config = None
        if api_key:
            auth_config = weaviate.auth.AuthApiKey(api_key=api_key)
        
        # Initialize client
        try:
            # Parse URL to get host and port
            parsed_url = url.replace("http://", "").replace("https://", "")
            host = parsed_url.split(":")[0] if ":" in parsed_url else parsed_url
            port = int(parsed_url.split(":")[-1]) if ":" in parsed_url else 8080
            
            # Connect to Weaviate
            self.client = weaviate.connect_to_local(
                host=host,
                port=port,
                # grpc_port=50051, # Uncomment if using gRPC
                auth_credentials=auth_config
            )

            # Test connection by getting meta info
            meta = self.client.get_meta()
            version = meta.get("version", "unknown")
            logger.info(f"Connected to Weaviate version {version} at {url}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}", exc_info=True)
            raise
    
    def set_llm_manager(self, llm_manager: LLMManager) -> None:
        """
        Set the LLM manager for generating embeddings.
        
        Args:
            llm_manager: LLM manager instance
        """
        self.llm_manager = llm_manager
    
    def create_collection_if_not_exists(self, collection_name: str) -> None:
        """
        Create a collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection to create
        """
    try:
        # Check if collection exists using the v4 client method
        if self.client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists.")
            return

        # Define properties using v4 classes
        properties = [
            wvc.config.Property(name="summary", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
            wvc.config.Property(name="keywords", data_type=wvc.config.DataType.TEXT_ARRAY, skip_vectorization=True),
            wvc.config.Property(name="critical_information", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
            wvc.config.Property(name="importance", data_type=wvc.config.DataType.INT, skip_vectorization=True),
            wvc.config.Property(name="timestamp", data_type=wvc.config.DataType.TEXT, skip_vectorization=True), # Consider DATE type if needed
            wvc.config.Property(name="location", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
            wvc.config.Property(name="agent", data_type=wvc.config.DataType.TEXT, skip_vectorization=True),
            # Weaviate generates UUIDs automatically, no need for an explicit 'id' property unless it's a custom ID
        ]

        # Create the collection with properties and vectorizer config
        self.client.collections.create(
            name=collection_name,
            description=f"Memory collection for {collection_name}",
            properties=properties,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(
                vectorize_collection_name=False
            ),
            # Define which property to vectorize (usually the main text content)
            vector_index_config=wvc.config.Configure.vector_index(
                 index_type=wvc.config.VectorIndexType.HNSW, # Or other types like FLAT
                 distance_metric=wvc.config.DistanceMetric.COSINE # Or other metrics
            )
        )
        logger.info(f"Created collection '{collection_name}'")

    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {e}", exc_info=True)
        raise
    
    def add_object(self, collection_name: str, properties: Dict[str, Any], vector_field: str = None) -> str:
        """
        Add an object to a collection.
        
        Args:
            collection_name: Name of the collection
            properties: Properties of the object
            vector_field: Field to use for vectorization (not used in newer API)
            
        Returns:
            ID of the created object
        """
        try:
            # Generate UUID if not provided
            obj_id = properties.get("id", str(uuid.uuid4()))
            
            # Get the collection
            collection = self.client.collections.get(collection_name)
            
            # Add the object
            collection.data.insert(properties, obj_id)
            
            logging.info(f"Added object to {collection_name} with ID {obj_id}")
            return obj_id
        
        except Exception as e:
            logging.error(f"Error adding object to {collection_name}: {e}")
            raise
    
    def get_object(self, collection_name: str, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an object by ID.
        
        Args:
            collection_name: Name of the collection
            object_id: ID of the object
            
        Returns:
            The object or None if not found
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)
            
            # Get the object by UUID
            # Note: object_id should be a valid UUID string or uuid.UUID object
            obj_uuid = uuid.UUID(object_id) if isinstance(object_id, str) else object_id
            result = collection.query.fetch_object_by_id(obj_uuid)

            if result:
                # Access properties directly from the object
                return result.properties
            return None
        except Exception as e:
            logging.error(f"Error getting object {object_id} from {collection_name}: {e}")
            return None
    
    def delete_object(self, collection_name: str, object_id: str) -> bool:
        """
        Delete an object by ID.
        
        Args:
            collection_name: Name of the collection
            object_id: ID of the object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)
            
            # Delete the object by UUID
            obj_uuid = uuid.UUID(object_id) if isinstance(object_id, str) else object_id
            collection.data.delete_by_id(obj_uuid)

            logger.info(f"Deleted object {object_id} from {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting object {object_id} from {collection_name}: {e}", exc_info=True)
            return False
    def semantic_search(self, collection_name: str, query: str, 
                        filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a semantic search.
        
        Args:
            collection_name: Name of the collection
            query: Query text
            filters: Optional filters
            limit: Maximum number of results
            
        Returns:
            List of matching objects
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)
            
            # Build the filter
            filter_query = self._build_filter_query(filters)
            
            # Execute the query using collection.query
            results = collection.query.near_text(
                query=query,
                limit=limit,
                filters=filter_query,
                return_metadata=wvc.query.MetadataQuery(distance=True) # Optional: get distance
            )

            # Convert results to dictionaries
            return [obj.properties for obj in results.objects]

        except Exception as e:
            logger.error(f"Error performing semantic search in {collection_name}: {e}", exc_info=True)
            return []
    
    def keyword_search(self, collection_name: str, query: str, 
                       filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a keyword search.
        
        Args:
            collection_name: Name of the collection
            query: Keyword query
            filters: Optional filters
            limit: Maximum number of results
            
        Returns:
            List of matching objects
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)
            
            # Build the filter
            filter_query = self._build_filter_query(filters)
            
            # Execute the query using collection.query
            results = collection.query.bm25(
                query=query,
                limit=limit,
                filters=filter_query,
                # query_properties=["summary^2", "keywords"] # Optional: specify properties and weights
            )

            # Convert results to dictionaries
            return [obj.properties for obj in results.objects]

        except Exception as e:
            logger.error(f"Error performing keyword search in {collection_name}: {e}", exc_info=True)
            return []
    
    def hybrid_search(self, collection_name: str, query: str, keyword_query: str = None,
                      filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search (semantic + keyword).
        
        Args:
            collection_name: Name of the collection
            query: Semantic query
            keyword_query: Keyword query (not used in newer API, query is used for both)
            filters: Optional filters
            limit: Maximum number of results
            
        Returns:
            List of matching objects
        """
        try:
            # Get the collection
            collection = self.client.collections.get(collection_name)
            
            # Build the filter
            filter_query = self._build_filter_query(filters)
            
            # Execute the query using collection.query
            results = collection.query.hybrid(
                query=query,
                alpha=0.5,  # Balance between vector and keyword search (0 = keyword, 1 = vector)
                limit=limit,
                filters=filter_query,
                # query_properties=["summary^2", "keywords"] # Optional: specify properties for keyword part
                return_metadata=wvc.query.MetadataQuery(score=True) # Optional: get hybrid score
            )

            # Convert results to dictionaries
            return [obj.properties for obj in results.objects]

        except Exception as e:
            logger.error(f"Error performing hybrid search in {collection_name}: {e}", exc_info=True)
            return []
    
    @staticmethod
    def _build_filter_query(filters: Optional[Dict[str, Any]]) -> Optional[weaviate.classes.query.Filter]:
        """
        Build a filter query for Weaviate queries.
        
        Args:
            filters: Filter specification
            
        Returns:
            Weaviate filter query
        """
        if not filters:
            return None
        
        filter_conditions = []
        
        # Process category filter
        if "category" in filters:
            filter_conditions.append(
                weaviate.classes.query.Filter.by_property("category").equal(filters["category"])
            )
        
        # Process importance filters
        if "min_importance" in filters:
            filter_conditions.append(
                weaviate.classes.query.Filter.by_property("importance").greater_than_equal(filters["min_importance"])
            )
        
        if "max_importance" in filters:
            filter_conditions.append(
                weaviate.classes.query.Filter.by_property("importance").less_than_equal(filters["max_importance"])
            )
        # Combine filters with AND
        if len(filter_conditions) > 1:
            # Use Filter.all_of for combining multiple filters
            return wvc.query.Filter.all_of(filter_conditions)
        elif len(filter_conditions) == 1:
            return filter_conditions[0]

        return None
