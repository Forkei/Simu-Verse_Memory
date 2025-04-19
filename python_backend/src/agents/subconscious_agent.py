import datetime
import uuid
import logging
import os
import yaml
from typing import Dict, List, Any, Optional, Union
import xml.etree.ElementTree as ET

# Setup logging
from ..utils.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from ..llm.llm_manager import LLMManager
from ..memory.weaviate_client import WeaviateClient
from ..memory.mock_weaviate_client import MockWeaviateClient

class SubconsciousAgent:
    """
    Subconscious agent that manages memories for a main agent.
    """
    
    def __init__(self, agent_name: str, llm_manager: LLMManager, 
                 weaviate_client: Union[WeaviateClient, MockWeaviateClient], memory_categories: Dict[str, Any]):
        """
        Initialize a subconscious agent.
        
        Args:
            agent_name: Name of the associated main agent
            llm_manager: LLM manager for generating responses
            weaviate_client: Weaviate client for memory storage
            memory_categories: Configuration for memory categories
        """
        self.agent_name = agent_name
        self.llm_manager = llm_manager
        self.weaviate_client = weaviate_client
        self.memory_categories = memory_categories
        self.collection_name = f"Memories_{agent_name}" # Ensure collection name is valid for Weaviate (e.g., starts with uppercase)
        self.collection_name = self.collection_name.replace("_", "").capitalize() # Basic sanitization

        # Load config first
        self.config = self._load_config()

        # System prompts loaded and formatted during initialization
        self.memory_creation_prompt = self._get_memory_creation_prompt()
        self.memory_retrieval_prompt = self._get_memory_retrieval_prompt()

        # Ensure the collection exists when the subconscious agent is created
        self.weaviate_client.create_collection_if_not_exists(self.collection_name)


    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml."""
        # Go up two levels from subconscious_agent.py to src/, then to config/
        base_path = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_path, "config", "config.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}")
            return {"paths": {}} # Return default empty paths
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file {config_path}: {e}")
            return {"paths": {}} # Return default empty paths

    def _get_config_path(self, key: str, default: str) -> str:
        """Helper to get a path from config, relative to src dir."""
        base_path = os.path.dirname(os.path.dirname(__file__)) # src directory
        relative_path = self.config.get("paths", {}).get(key, default)
        return os.path.join(base_path, relative_path)

    def _load_prompt_template(self, template_name: str) -> str:
        """Loads a prompt template from the templates directory using path from config."""
        templates_dir = self._get_config_path("prompt_templates", "agents/templates/")
        template_path = os.path.join(templates_dir, template_name)
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"Prompt template file not found: {template_path}")
            return "" # Return empty string or raise an error

    def _get_memory_creation_prompt(self) -> str:
        """Get the system prompt for memory creation by loading and formatting a template."""
        template = self._load_prompt_template("subconscious_memory_creation_prompt_template.txt")
        if not template:
            return "Error: Memory creation prompt template missing." # Fallback

        categories = ", ".join([cat["name"] for cat in self.memory_categories["categories"]])
        prompt = template.replace("{{AGENT_NAME}}", self.agent_name)
        prompt = prompt.replace("{{MEMORY_CATEGORIES}}", categories)
        return prompt

    def _get_memory_retrieval_prompt(self) -> str:
        """Get the system prompt for memory retrieval by loading and formatting a template."""
        template = self._load_prompt_template("subconscious_memory_retrieval_prompt_template.txt")
        if not template:
            return "Error: Memory retrieval prompt template missing." # Fallback

        prompt = template.replace("{{AGENT_NAME}}", self.agent_name)
        return prompt

    def create_memory_from_conversation(self, conversation: List[Dict[str, str]], location: str) -> Optional[Dict[str, Any]]:
        """
        Create a memory from recent conversation.
        
        Args:
            conversation: List of conversation messages
            location: Current location of the agent
            
        Returns:
            The created memory
        """
        # Format conversation for the LLM
        conversation_text = ""
        for message in conversation:
            role = "Agent" if message["role"] == "assistant" else "User"
            conversation_text += f"{role}: {message['content']}\n\n"
        # --- Generate Memory with Retry Logic ---
        memory_xml = None
        memory = None
        last_error = None

        for attempt in range(MAX_LLM_RETRIES):
            logger.debug(f"Attempt {attempt + 1}/{MAX_LLM_RETRIES} to generate memory for {self.agent_name}...")
            # Format conversation for the LLM each time, potentially adding error context
            conversation_text = ""
            for message in conversation:
                role = "Agent" if message["role"] == "assistant" else "User"
                conversation_text += f"{role}: {message['content']}\n\n"
            
            prompt = f"Create a memory from this conversation at location '{location}':\n\n{conversation_text}"
            if last_error:
                prompt += f"\n\n[SYSTEM_ERROR: Previous attempt failed: {last_error}. Please ensure response is valid XML with a single <memory> root tag and all required child tags (summary, category, keywords, critical_information, importance).]"

            try:
                memory_xml = self.llm_manager.generate_response(prompt, self.memory_creation_prompt)
                logger.debug(f"Raw memory XML attempt {attempt + 1} from LLM for {self.agent_name}: {memory_xml[:150]}...")

                # Parse the XML response
                memory = self._parse_memory_xml(memory_xml)
                if not memory or not memory.get("summary"): # Ensure at least summary is present
                    raise ValueError("Parsed memory XML is invalid or missing summary.")

                # If successful, break the loop
                logger.info(f"Successfully generated and parsed memory for {self.agent_name} on attempt {attempt + 1}.")
                last_error = None
                break

            except Exception as e:
                last_error = e
                logger.warning(f"Memory creation attempt {attempt + 1} failed for {self.agent_name}: {e}. Raw XML: {memory_xml}")
                if attempt < MAX_LLM_RETRIES - 1:
                    logger.info(f"Retrying memory creation after {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.error(f"Agent {self.agent_name} failed to generate valid memory XML after {MAX_LLM_RETRIES} attempts.")
                    return None # Failed to create memory

        # If memory parsing failed after retries, return None
        if last_error is not None:
            return None

        # --- Add Metadata and Store ---
        memory["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat() # Use UTC
        memory["location"] = location
        memory["agent"] = self.agent_name
        memory["id"] = str(uuid.uuid4()) # Generate UUID here

        # Store in Weaviate
        try:
            # Pass properties without the 'id' key if Weaviate generates it automatically
            properties_to_add = {k: v for k, v in memory.items() if k != 'id'}

            inserted_uuid = self.weaviate_client.add_object(
                collection_name=self.collection_name,
                properties=properties_to_add,
                uuid=memory["id"] # Pass the generated UUID to Weaviate
            )
            logger.info(f"Memory stored for {self.agent_name}. Weaviate ID: {inserted_uuid}")
            return memory # Return the complete memory dict including the ID we generated
        except Exception as e:
            logger.error(f"Failed to add memory object to Weaviate for {self.agent_name}: {e}", exc_info=True)
            return None # Return None on storage failure

    def _parse_memory_xml(self, xml_string: str) -> Optional[Dict[str, Any]]:
        """
        Parse the XML response from the LLM into a memory object.
        
        """
        Parse the XML response from the LLM into a memory object.

        Args:
            xml_string: XML-formatted memory from LLM

        Returns:
            Dictionary representation of the memory, or None if parsing fails
        """
        try:
            # Clean potential markdown code blocks and whitespace
            cleaned_xml = xml_string.strip()
            if cleaned_xml.startswith("```xml"):
                cleaned_xml = cleaned_xml[len("```xml"):].strip()
            if cleaned_xml.endswith("```"):
                cleaned_xml = cleaned_xml[:-len("```")].strip()

            # Find the <memory> tag, ignore anything outside it
            start_tag = "<memory>"
            end_tag = "</memory>"
            start_index = cleaned_xml.find(start_tag)
            end_index = cleaned_xml.rfind(end_tag)

            if start_index == -1 or end_index == -1:
                logger.warning(f"Could not find <memory>...</memory> tags in response: {cleaned_xml[:200]}...")
                raise ValueError("Missing <memory> tags")

            memory_content = cleaned_xml[start_index : end_index + len(end_tag)]

            # Parse the extracted <memory> element
            root = ET.fromstring(memory_content) # Should be <memory> element

            if root.tag != 'memory':
                 logger.warning(f"Root tag is not <memory>: {root.tag}")
                 raise ValueError("Root tag is not <memory>")

            memory = {}
            required_fields = ["summary", "category", "keywords", "critical_information", "importance"]
            valid_categories = [cat["name"] for cat in self.memory_categories.get("categories", [])]

            for child in root:
                field = child.tag
                value = child.text.strip() if child.text else ""

                if field == "importance":
                    try:
                        imp_val = int(value)
                        # Clamp importance between 1 and 10
                        memory[field] = max(1, min(10, imp_val))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid importance value '{value}', using default 5.")
                        memory[field] = 5 # Default importance
                elif field == "keywords":
                    memory[field] = [k.strip() for k in value.split(",") if k.strip()]
                elif field == "category":
                     if value in valid_categories:
                         memory[field] = value
                     else:
                         logger.warning(f"Invalid memory category '{value}', using 'observation'. Valid: {valid_categories}")
                         memory[field] = "observation" # Default category
                else:
                    memory[field] = value

            # Check if all required fields are present after parsing
            missing_fields = [f for f in required_fields if f not in memory]
            if missing_fields:
                logger.warning(f"Parsed memory XML is missing required fields: {missing_fields}. XML: {memory_content}")
                # Allow partial memory only if summary exists
                if "summary" not in memory or not memory["summary"]:
                     raise ValueError(f"Missing required fields: {missing_fields}")
                # Fill missing fields with defaults if summary exists
                for field in missing_fields:
                     if field == "importance": memory[field] = 5
                     elif field == "keywords": memory[field] = []
                     elif field == "category": memory[field] = "observation"
                     else: memory[field] = ""


            return memory
        except ET.ParseError as e:
            logger.error(f"XML parsing error for memory: {e}\nInvalid XML: {xml_string[:500]}...", exc_info=False) # Log less verbosely
            raise ValueError(f"Invalid XML format: {e}") # Re-raise for retry logic
        except Exception as e:
            logger.error(f"Unexpected error parsing memory XML: {e}\nXML: {xml_string[:500]}...", exc_info=True)
            raise ValueError(f"Unexpected error parsing XML: {e}") # Re-raise for retry logic


    def retrieve_relevant_memories(self, conversation: List[Dict[str, str]], location: str) -> List[Dict[str, Any]]:
            return None


    def retrieve_relevant_memories(self, conversation: List[Dict[str, str]], location: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on current context.
        
        Args:
            conversation: Recent conversation history
            location: Current location of the agent
            
        Returns:
            List of relevant memories
        """
        # If no conversation, return empty list
        if not conversation:
            return []
        
        # Format conversation for the LLM
        conversation_text = ""
        for message in conversation[-5:]:  # Use last 5 messages
            role = "Agent" if message["role"] == "assistant" else "User"
            conversation_text += f"{role}: {message['content']}\n\n"
        # --- Generate Memory Queries with Retry Logic ---
        queries_xml = None
        queries = None
        last_error = None

        for attempt in range(MAX_LLM_RETRIES):
             logger.debug(f"Attempt {attempt + 1}/{MAX_LLM_RETRIES} to generate memory queries for {self.agent_name}...")
             # Format conversation for the LLM each time
             conversation_text = ""
             for message in conversation[-5:]:  # Use last 5 messages
                 role = "Agent" if message["role"] == "assistant" else "User"
                 conversation_text += f"{role}: {message['content']}\n\n"

             prompt = f"Create memory queries based on this conversation at location '{location}':\n\n{conversation_text}"
             if last_error:
                 prompt += f"\n\n[SYSTEM_ERROR: Previous attempt failed: {last_error}. Please ensure response is valid XML with a <memory_queries> root tag containing one or more <query> tags.]"

             try:
                 queries_xml = self.llm_manager.generate_response(prompt, self.memory_retrieval_prompt)
                 logger.debug(f"Raw memory queries XML attempt {attempt + 1} from LLM for {self.agent_name}: {queries_xml[:150]}...")

                 # Parse the XML response to get queries
                 queries = self._parse_memory_queries_xml(queries_xml)
                 if not queries: # Check if parsing returned any queries
                      raise ValueError("Parsed memory queries XML is invalid or empty.")

                 # If successful, break the loop
                 logger.info(f"Successfully generated and parsed memory queries for {self.agent_name} on attempt {attempt + 1}.")
                 last_error = None
                 break

             except Exception as e:
                 last_error = e
                 logger.warning(f"Memory query generation attempt {attempt + 1} failed for {self.agent_name}: {e}. Raw XML: {queries_xml}")
                 if attempt < MAX_LLM_RETRIES - 1:
                     logger.info(f"Retrying memory query generation after {RETRY_DELAY_SECONDS} seconds...")
                     time.sleep(RETRY_DELAY_SECONDS)
                 else:
                     logger.error(f"Agent {self.agent_name} failed to generate valid memory queries XML after {MAX_LLM_RETRIES} attempts.")
                     return [] # Failed to generate queries

        # If query generation failed after retries, return empty list
        if last_error is not None:
            return []

        # --- Execute Queries and Process Results ---
        all_memories = []
        for query_data in queries:
            try:
                memories = self._execute_memory_query(query_data)
                if memories:
                    all_memories.extend(memories)
            except Exception as e:
                 logger.error(f"Error executing memory query for {self.agent_name}: {query_data}. Error: {e}", exc_info=True)

        # Remove duplicates (based on memory ID - Weaviate UUID)
        unique_memories = {}
        for memory in all_memories:
            mem_id = memory.get("id") # Weaviate objects have 'id' in properties if fetched correctly
            if mem_id and mem_id not in unique_memories:
                unique_memories[mem_id] = memory

        # Sort by importance (descending) and return top results
        sorted_memories = sorted(unique_memories.values(), key=lambda x: x.get("importance", 0), reverse=True)
        
        # Limit the number of memories returned (e.g., top 5 overall)
        final_memories = sorted_memories[:5]
        logger.info(f"Retrieved {len(final_memories)} unique memories for {self.agent_name} after processing {len(queries)} queries.")
        return final_memories

    def _parse_memory_queries_xml(self, xml_string: str) -> List[Dict[str, Any]]:
        """
        Parse the XML response from the LLM into memory queries.
        
        """
        Parse the XML response from the LLM into memory queries.

        Args:
            xml_string: XML-formatted memory queries from LLM

        Returns:
            List of query dictionaries, max 3.
        """
        queries = []
        try:
            # Clean potential markdown code blocks and whitespace
            cleaned_xml = xml_string.strip()
            if cleaned_xml.startswith("```xml"):
                cleaned_xml = cleaned_xml[len("```xml"):].strip()
            if cleaned_xml.endswith("```"):
                cleaned_xml = cleaned_xml[:-len("```")].strip()

            # Find the <memory_queries> tag, ignore anything outside it
            start_tag = "<memory_queries>"
            end_tag = "</memory_queries>"
            start_index = cleaned_xml.find(start_tag)
            end_index = cleaned_xml.rfind(end_tag)

            if start_index == -1 or end_index == -1:
                logger.warning(f"Could not find <memory_queries>...</memory_queries> tags in response: {cleaned_xml[:200]}...")
                raise ValueError("Missing <memory_queries> tags")

            queries_content = cleaned_xml[start_index : end_index + len(end_tag)]

            root = ET.fromstring(queries_content) # Should be <memory_queries>

            if root.tag != 'memory_queries':
                 logger.warning(f"Root tag is not <memory_queries>: {root.tag}")
                 raise ValueError("Root tag is not <memory_queries>")

            for query_element in root.findall('.//query'):
                if len(queries) >= 3: # Limit to 3 queries max
                    logger.warning("LLM provided more than 3 queries, limiting to first 3.")
                    break

                query = {}
                filters = {}
                for child in query_element:
                    tag = child.tag
                    value = child.text.strip() if child.text else ""

                    if tag == "filters":
                        for filter_child in child:
                            field = filter_child.tag
                            filter_value = filter_child.text.strip() if filter_child.text else ""
                            if field in ["min_importance", "max_importance"]:
                                try:
                                    filters[field] = int(filter_value)
                                except (ValueError, TypeError):
                                    logger.warning(f"Invalid importance filter value '{filter_value}' in query XML.")
                            elif field == "category":
                                 # Validate category if needed, or pass through
                                 filters[field] = filter_value
                            # Add other potential filters here (datetime, metadata) if needed later
                            else:
                                 logger.warning(f"Unknown filter field '{field}' in query XML.")
                    elif tag == "keywords":
                        # Handle potential empty keywords tag
                        query[tag] = [k.strip() for k in value.split(",") if k.strip()] if value else []
                    elif tag in ["search_type", "query_text"]:
                        query[tag] = value
                    else:
                        logger.warning(f"Unknown tag '{tag}' inside <query> element.")

                # Validate required fields for a query
                search_type = query.get("search_type")
                if not search_type or search_type not in ["keyword", "semantic", "hybrid"]:
                    logger.warning(f"Invalid or missing search_type in query: {query}. Skipping query.")
                    continue
                if search_type in ["semantic", "hybrid"] and not query.get("query_text"):
                    logger.warning(f"Missing query_text for {search_type} search. Skipping query: {query}")
                    continue
                if search_type in ["keyword", "hybrid"] and not query.get("keywords"):
                    logger.warning(f"Missing keywords for {search_type} search. Skipping query: {query}")
                    continue


                if filters:
                    query["filters"] = filters
                queries.append(query)

            logger.debug(f"Parsed {len(queries)} valid memory queries.")
            return queries

        except ET.ParseError as e:
            logger.error(f"XML parsing error for memory queries: {e}\nInvalid XML: {xml_string[:500]}...", exc_info=False)
            raise ValueError(f"Invalid XML format for queries: {e}") # Re-raise for retry logic
        except Exception as e:
            logger.error(f"Unexpected error parsing memory queries XML: {e}\nXML: {xml_string[:500]}...", exc_info=True)
            raise ValueError(f"Unexpected error parsing query XML: {e}") # Re-raise for retry logic

    def _execute_memory_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
            return []

    def _execute_memory_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a memory query against the Weaviate database.
        
        Args:
            query: Query dictionary
            
        Returns:
            List of memory objects matching the query
        """
        search_type = query.get("search_type", "hybrid")
        filters = query.get("filters", {})
        
        # Build Weaviate filter
        weaviate_filter = {}
        # Build Weaviate filter using v4 Filter class
        weaviate_filters_list = []
        if "category" in filters:
            weaviate_filters_list.append(
                 wvc.query.Filter.by_property("category").equal(filters["category"])
            )
        if "min_importance" in filters:
             weaviate_filters_list.append(
                 wvc.query.Filter.by_property("importance").greater_than_equal(filters["min_importance"])
             )
        if "max_importance" in filters:
             weaviate_filters_list.append(
                 wvc.query.Filter.by_property("importance").less_than_equal(filters["max_importance"])
             )
        # Add other filters (datetime, metadata) here when implemented

        # Combine filters using AND logic if multiple exist
        final_filter = None
        if len(weaviate_filters_list) > 1:
            final_filter = wvc.query.Filter.all_of(weaviate_filters_list)
        elif len(weaviate_filters_list) == 1:
            final_filter = weaviate_filters_list[0]

        logger.debug(f"Executing memory query for {self.agent_name}: Type={search_type}, Filters={filters}, WeaviateFilter={final_filter}")

        # Execute the appropriate search based on search type
        try:
            if search_type == "keyword":
                # Keyword search
                keywords = query.get("keywords", [])
                if not keywords:
                    logger.warning("Keyword search requested but no keywords provided.")
                    return []

                # Join keywords with spaces for BM25
                keyword_query_str = " ".join(keywords)
                logger.debug(f"Keyword query string: '{keyword_query_str}'")

                return self.weaviate_client.keyword_search(
                    collection_name=self.collection_name,
                    query=keyword_query_str,
                    filters=final_filter, # Pass the constructed filter object
                    limit=3 # Limit results per query type
                )

            elif search_type == "semantic":
                # Semantic search
                query_text = query.get("query_text", "")
                if not query_text:
                    logger.warning("Semantic search requested but no query text provided.")
                    return []
                logger.debug(f"Semantic query text: '{query_text}'")

                return self.weaviate_client.semantic_search(
                    collection_name=self.collection_name,
                    query=query_text,
                    filters=final_filter, # Pass the constructed filter object
                    limit=3
                )

            else:  # hybrid
                # Hybrid search (both keyword and semantic)
                query_text = query.get("query_text", "")
                keywords = query.get("keywords", [])
                keyword_query_str = " ".join(keywords) # For logging/potential future use

                # Weaviate v4 hybrid uses the single 'query' param for both semantic vector and BM25 text
                if not query_text:
                     logger.warning("Hybrid search requires query_text. Skipping.")
                     return []

                logger.debug(f"Hybrid query text: '{query_text}' (Keywords used by BM25: '{keyword_query_str}')")

                return self.weaviate_client.hybrid_search(
                    collection_name=self.collection_name,
                    query=query_text, # Used for both semantic and keyword parts
                    # keyword_query=keyword_query_str, # Not used in v4 hybrid call directly
                    filters=final_filter, # Pass the constructed filter object
                    limit=3
                )
        except Exception as search_error:
             logger.error(f"Error during Weaviate search ({search_type}) for agent {self.agent_name}: {search_error}", exc_info=True)
            # Keyword search
            keywords = query.get("keywords", [])
            if not keywords:
                return []
            
            # Join keywords with OR for broader search
            keyword_query = " OR ".join(keywords)
            logger.debug(f"Keyword query: {keyword_query}")

            return self.weaviate_client.keyword_search(
                collection_name=self.collection_name,
                query=keyword_query,
                filters=weaviate_filter,
                limit=3
            )
        
        elif search_type == "semantic":
            # Semantic search
            query_text = query.get("query_text", "")
            if not query_text:
                logger.warning("Semantic search requested but no query text provided.")
                return []
            logger.debug(f"Semantic query: {query_text}")

            return self.weaviate_client.semantic_search(
                collection_name=self.collection_name,
                query=query_text,
                filters=weaviate_filter,
                limit=3
            )
        
        else:  # hybrid
            # Hybrid search (both keyword and semantic)
            query_text = query.get("query_text", "")
            keywords = query.get("keywords", [])
            
            if not query_text and not keywords:
                return []

            if query_text and keywords:
                # If we have both, do a hybrid search
                keyword_query = " OR ".join(keywords)
                logger.debug(f"Hybrid query: Semantic='{query_text}', Keywords='{keyword_query}'")

                return self.weaviate_client.hybrid_search(
                    collection_name=self.collection_name,
                    query=query_text,
                    keyword_query=keyword_query, # Note: keyword_query might not be used in v4 hybrid
                    filters=weaviate_filter,
                    limit=3
                )
            elif query_text:
                # If we only have query text, do semantic search
                logger.debug(f"Hybrid query (semantic only): '{query_text}'")
                return self.weaviate_client.semantic_search(
                    collection_name=self.collection_name,
                    query=query_text,
                    filters=weaviate_filter,
                    limit=3
                )
            else: # Only keywords
                # If we only have keywords, do keyword search
                keyword_query = " OR ".join(keywords)
                logger.debug(f"Hybrid query (keyword only): '{keyword_query}'")

                return self.weaviate_client.keyword_search(
                    collection_name=self.collection_name,
                    query=keyword_query,
                    filters=weaviate_filter,
                    limit=3
                )
