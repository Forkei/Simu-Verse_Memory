import uuid
import logging
from typing import Dict, List, Any, Optional
import weaviate
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
                auth_credentials=auth_config
            )
            
            # Test connection by getting meta info
            meta = self.client.get_meta()
            version = meta.get("version", "unknown")
            logging.info(f"Connected to Weaviate version {version} at {url}")
        except Exception as e:
            logging.error(f"Failed to connect to Weaviate: {e}")
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
        # Check if collection exists
        try:
            # Check if collection exists
            collections = self.client.collections.list_all()
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                logging.info(f"Collection {collection_name} already exists")
                return
            
            # Create the collection with properties
            collection = self.client.collections.create(
                name=collection_name,
                description=f"Memory collection for {collection_name}",
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_transformers(
                    pooling_strategy="masked_mean",
                    vectorize_collection_name=False
                )
            )
            
            # Add properties to the collection
            collection.properties.create(
                name="summary",
                description="Summary of the memory",
                data_type=weaviate.classes.config.DataType.TEXT,
                skip_vectorization=False
            )
            
            collection.properties.create(
                name="category",
                description="Category of the memory",
                data_type=weaviate.classes.config.DataType.TEXT,
                skip_vectorization=True
            )
            
            collection.properties.create(
                name="keywords",
                description="Keywords related to the memory",
                data_type=weaviate.classes.config.DataType.TEXT_ARRAY,
                skip_vectorization=True
            )
            
            collection.properties.create(
                name="critical_information",
                description="Critical information in the memory",
                data_type=weaviate.classes.config.DataType.TEXT,
                skip_vectorization=True
            )
            
            collection.properties.create(
                name="importance",
                description="Importance of the memory (1-10)",
                data_type=weaviate.classes.config.DataType.INT,
                skip_vectorization=True
            )
            
            collection.properties.create(
                name="timestamp",
                description="Timestamp of the memory",
                data_type=weaviate.classes.config.DataType.TEXT,
                skip_vectorization=True
            )
            
            collection.properties.create(
                name="location",
                description="Location where the memory was created",
                data_type=weaviate.classes.config.DataType.TEXT,
                skip_vectorization=True
            )
            
            collection.properties.create(
                name="agent",
                description="Agent who owns the memory",
                data_type=weaviate.classes.config.DataType.TEXT,
                skip_vectorization=True
            )
            
            collection.properties.create(
                name="id",
                description="Unique ID of the memory",
                data_type=weaviate.classes.config.DataType.TEXT,
                skip_vectorization=True
            )
            logging.info(f"Created collection {collection_name}")
        
        except Exception as e:
            logging.error(f"Error creating collection {collection_name}: {e}")
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
            
            # Get the object
            result = collection.query.fetch_object_by_id(object_id)
            
            if result:
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
            
            # Delete the object
            collection.data.delete_by_id(object_id)
            
            logging.info(f"Deleted object {object_id} from {collection_name}")
            return True
        
        except Exception as e:
            logging.error(f"Error deleting object {object_id} from {collection_name}: {e}")
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
            
            # Execute the query
            results = collection.query.near_text(
                query=query,
                limit=limit,
                filters=filter_query
            )
            
            # Convert results to dictionaries
            return [obj.properties for obj in results.objects]
        
        except Exception as e:
            logging.error(f"Error performing semantic search in {collection_name}: {e}")
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
            
            # Execute the query
            results = collection.query.bm25(
                query=query,
                limit=limit,
                filters=filter_query
            )
            
            # Convert results to dictionaries
            return [obj.properties for obj in results.objects]
        
        except Exception as e:
            logging.error(f"Error performing keyword search in {collection_name}: {e}")
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
            
            # Execute the query
            results = collection.query.hybrid(
                query=query,
                alpha=0.5,  # Balance between vector and keyword search
                limit=limit,
                filters=filter_query
            )
            
            # Convert results to dictionaries
            return [obj.properties for obj in results.objects]
        
        except Exception as e:
            logging.error(f"Error performing hybrid search in {collection_name}: {e}")
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
            return weaviate.classes.query.Filter.all_of(*filter_conditions)
        elif len(filter_conditions) == 1:
            return filter_conditions[0]
        
        return None
