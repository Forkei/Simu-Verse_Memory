import os
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, Union
import weaviate
from weaviate.exceptions import WeaviateQueryError
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
            self.client = weaviate.Client(
                url=url,
                auth_client_secret=auth_config
            )
            logging.info(f"Connected to Weaviate at {url}")
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
            schema = self.client.schema.get()
            classes = [c["class"] for c in schema["classes"]] if "classes" in schema else []
            
            if collection_name in classes:
                logging.info(f"Collection {collection_name} already exists")
                return
            
            # Define the class/collection
            class_obj = {
                "class": collection_name,
                "description": f"Memory collection for {collection_name}",
                "vectorizer": "text2vec-transformers",  # Using transformers for vectorization
                "moduleConfig": {
                    "text2vec-transformers": {
                        "poolingStrategy": "masked_mean",
                        "vectorizeClassName": false
                    }
                },
                "properties": [
                    {
                        "name": "summary",
                        "description": "Summary of the memory",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "category",
                        "description": "Category of the memory",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "keywords",
                        "description": "Keywords related to the memory",
                        "dataType": ["text[]"],
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": True,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "critical_information",
                        "description": "Critical information in the memory",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": True,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "importance",
                        "description": "Importance of the memory (1-10)",
                        "dataType": ["int"],
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": True,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "timestamp",
                        "description": "Timestamp of the memory",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": True,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "location",
                        "description": "Location where the memory was created",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": True,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "agent",
                        "description": "Agent who owns the memory",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": True,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "id",
                        "description": "Unique ID of the memory",
                        "dataType": ["text"],
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": True,
                                "vectorizePropertyName": False
                            }
                        }
                    }
                ]
            }
            
            # Create the class
            self.client.schema.create_class(class_obj)
            logging.info(f"Created collection {collection_name}")
        
        except Exception as e:
            logging.error(f"Error creating collection {collection_name}: {e}")
            raise
    
    def add_object(self, collection_name: str, properties: Dict[str, Any], vector_field: str = "summary") -> str:
        """
        Add an object to a collection.
        
        Args:
            collection_name: Name of the collection
            properties: Properties of the object
            vector_field: Field to use for vectorization
            
        Returns:
            ID of the created object
        """
        try:
            # Generate UUID if not provided
            obj_id = properties.get("id", str(uuid.uuid4()))
            
            # Add the object
            self.client.data_object.create(
                data_object=properties,
                class_name=collection_name,
                uuid=obj_id
            )
            
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
            result = self.client.data_object.get_by_id(
                uuid=object_id,
                class_name=collection_name
            )
            
            if result:
                return result
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
            self.client.data_object.delete(
                uuid=object_id,
                class_name=collection_name
            )
            
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
            # Build the query
            where_filter = self._build_where_filter(filters)
            
            # Execute the query
            result = (
                self.client.query
                .get(collection_name, ["summary", "category", "keywords", "critical_information", 
                                       "importance", "timestamp", "location", "agent", "id"])
                .with_near_text({"concepts": [query]})
                .with_where(where_filter) if where_filter else self.client.query.get(collection_name)
                .with_limit(limit)
                .do()
            )
            
            # Extract and return the objects
            if "data" in result and "Get" in result["data"] and collection_name in result["data"]["Get"]:
                return result["data"]["Get"][collection_name]
            return []
        
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
            # Build the query
            where_filter = self._build_where_filter(filters)
            
            # Execute the query
            result = (
                self.client.query
                .get(collection_name, ["summary", "category", "keywords", "critical_information", 
                                       "importance", "timestamp", "location", "agent", "id"])
                .with_bm25({"query": query})
                .with_where(where_filter) if where_filter else self.client.query.get(collection_name)
                .with_limit(limit)
                .do()
            )
            
            # Extract and return the objects
            if "data" in result and "Get" in result["data"] and collection_name in result["data"]["Get"]:
                return result["data"]["Get"][collection_name]
            return []
        
        except Exception as e:
            logging.error(f"Error performing keyword search in {collection_name}: {e}")
            return []
    
    def hybrid_search(self, collection_name: str, query: str, keyword_query: str,
                      filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search (semantic + keyword).
        
        Args:
            collection_name: Name of the collection
            query: Semantic query
            keyword_query: Keyword query
            filters: Optional filters
            limit: Maximum number of results
            
        Returns:
            List of matching objects
        """
        try:
            # Build the query
            where_filter = self._build_where_filter(filters)
            
            # Execute the query
            result = (
                self.client.query
                .get(collection_name, ["summary", "category", "keywords", "critical_information", 
                                       "importance", "timestamp", "location", "agent", "id"])
                .with_hybrid({"query": query, "alpha": 0.5})  # Alpha balances between vector and keyword search
                .with_where(where_filter) if where_filter else self.client.query.get(collection_name)
                .with_limit(limit)
                .do()
            )
            
            # Extract and return the objects
            if "data" in result and "Get" in result["data"] and collection_name in result["data"]["Get"]:
                return result["data"]["Get"][collection_name]
            return []
        
        except Exception as e:
            logging.error(f"Error performing hybrid search in {collection_name}: {e}")
            return []
    
    def _build_where_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Build a where filter for Weaviate queries.
        
        Args:
            filters: Filter specification
            
        Returns:
            Weaviate where filter
        """
        if not filters:
            return None
        
        # Simple filter with a single condition
        if "path" in filters and "operator" in filters:
            return filters
        
        # Multiple filters
        operands = []
        
        # Process category filter
        if "category" in filters:
            operands.append({
                "path": ["category"],
                "operator": "Equal",
                "valueString": filters["category"]
            })
        
        # Process importance filters
        if "min_importance" in filters:
            operands.append({
                "path": ["importance"],
                "operator": "GreaterThanEqual",
                "valueNumber": filters["min_importance"]
            })
        
        if "max_importance" in filters:
            operands.append({
                "path": ["importance"],
                "operator": "LessThanEqual",
                "valueNumber": filters["max_importance"]
            })
        
        # If we have multiple operands, combine them with AND
        if len(operands) > 1:
            return {
                "operator": "And",
                "operands": operands
            }
        elif len(operands) == 1:
            return operands[0]
        
        return None
