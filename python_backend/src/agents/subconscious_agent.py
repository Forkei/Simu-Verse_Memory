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
        self.collection_name = f"Memories_{agent_name}"
        
        # System prompt for memory creation
        self.memory_creation_prompt = self._get_memory_creation_prompt()
        
        # Load config first
        self.config = self._load_config()

        # System prompt for memory creation
        self.memory_creation_prompt = self._get_memory_creation_prompt()

        # System prompt for memory retrieval
        self.memory_retrieval_prompt = self._get_memory_retrieval_prompt()

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

        # The original prompt text is now moved to the template file
        # prompt = f"""You are the subconscious mind of {self.agent_name}. Your task is to create a memory from recent conversation.
        # 
# Analyze the conversation and create a memory with the following components:
# 1. Summary: A concise description of what happened
# 2. Category: Choose from: {categories}
# 3. Keywords: 3-5 relevant keywords for future retrieval
# 4. Critical Information: Details that will help make good decisions when this memory is recalled
# 5. Importance: Rate from 1-10 how important this memory is (1 = trivial, 10 = life-changing)
# 
# Respond in XML format like this:
# <memory>
#   <summary>Brief summary of what happened</summary>
#   <category>one_of_the_categories</category>
#   <keywords>keyword1, keyword2, keyword3</keywords>
#   <critical_information>Important details that should influence future decisions</critical_information>
#   <importance>7</importance>
# </memory>
        return prompt

    def _get_memory_retrieval_prompt(self) -> str:
        """Get the system prompt for memory retrieval by loading and formatting a template."""
        template = self._load_prompt_template("subconscious_memory_retrieval_prompt_template.txt")
        if not template:
            return "Error: Memory retrieval prompt template missing." # Fallback

        prompt = template.replace("{{AGENT_NAME}}", self.agent_name)

        # The original prompt text is now moved to the template file
        # prompt = f"""You are the subconscious mind of {self.agent_name}. Your task is to create queries to retrieve relevant memories.
        #
        # Based on the recent conversation and current context, create up to 3 memory queries that will help retrieve the most relevant memories.
# 
# For each query, specify:
# 1. Search type: "keyword", "semantic", or "hybrid"
# 2. Keywords: If using keyword or hybrid search
# 3. Query text: For semantic or hybrid search
# 4. Filters: Any filters to apply (category, min/max importance, time range)
# 
# Respond in XML format like this:
# <memory_queries>
#   <query>
#     <search_type>hybrid</search_type>
#     <keywords>meeting, project, deadline</keywords>
#     <query_text>Important information about the project deadline</query_text>
#     <filters>
#       <category>conversation</category>
#       <min_importance>5</min_importance>
#     </filters>
#   </query>
#   <!-- Additional queries as needed, up to 3 total -->
# </memory_queries>
# """
        return prompt
    
    def create_memory_from_conversation(self, conversation: List[Dict[str, str]], location: str) -> Dict[str, Any]:
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
        
        # Generate memory using LLM
        prompt = f"Create a memory from this conversation at location '{location}':\n\n{conversation_text}"
        logger.debug(f"Generating memory creation prompt for {self.agent_name}")
        memory_xml = self.llm_manager.generate_response(prompt, self.memory_creation_prompt)
        logger.debug(f"Received memory XML from LLM for {self.agent_name}: {memory_xml[:100]}...")

        # Parse the XML response
        memory = self._parse_memory_xml(memory_xml)
        if not memory:
             logger.error(f"Failed to parse memory XML for {self.agent_name}. XML: {memory_xml}")
             return {} # Return empty dict if parsing failed

        # Add timestamp and location
        memory["timestamp"] = datetime.datetime.now().isoformat()
        memory["location"] = location
        memory["agent"] = self.agent_name
        memory["id"] = str(uuid.uuid4())
        
        # Store in Weaviate
        self.weaviate_client.add_object(
            collection_name=collection_name,
            properties=properties,
            uuid=obj_id # Pass the generated or existing UUID
            # vector=... # Optionally provide pre-computed vector
        )

        # The insert method now returns the UUID directly
        inserted_uuid = collection.data.insert(properties, uuid=obj_id)

        logger.info(f"Added object to {collection_name} with ID {inserted_uuid}")
        return str(inserted_uuid) # Return as string

    def _parse_memory_xml(self, xml_response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the XML response from the LLM into a memory object.
        
        Args:
            xml_response: XML-formatted memory from LLM
            
        Returns:
            Dictionary representation of the memory, or None if parsing fails
        """
        try:
            # Ensure the response is wrapped in a root element for valid XML parsing
            if not xml_response.strip().startswith('<'):
                 # Attempt to find the first < and last > if response has extra text
                 start = xml_response.find('<memory>')
                 end = xml_response.rfind('</memory>')
                 if start != -1 and end != -1:
                     xml_response = xml_response[start:end+len('</memory>')]
                 else: # Give up if no memory tags found
                     logger.warning(f"Invalid XML structure for memory parsing: {xml_response}")
                     return None

            # Add a dummy root if necessary (ElementTree needs a single root)
            if not xml_response.strip().startswith('<root>'):
                 xml_response = f"<root>{xml_response}</root>"

            root = ET.fromstring(xml_response)
            memory_element = root.find('memory')

            if memory_element is None:
                 logger.warning(f"Could not find <memory> tag in response: {xml_response}")
                 return None

            memory = {}
            for child in memory_element:
                field = child.tag
                value = child.text.strip() if child.text else ""

                if field == "importance":
                    try:
                        memory[field] = int(value)
                    except ValueError:
                        logger.warning(f"Invalid importance value '{value}', using default 5.")
                        memory[field] = 5
                elif field == "keywords":
                    memory[field] = [k.strip() for k in value.split(",") if k.strip()]
                else:
                    memory[field] = value

            # Basic validation
            if not all(k in memory for k in ["summary", "category", "keywords", "critical_information", "importance"]):
                 logger.warning(f"Parsed memory XML is missing required fields: {memory}")
                 # Allow partial memory if summary exists
                 if "summary" not in memory:
                     return None

            return memory
        except ET.ParseError as e:
            logger.error(f"XML parsing error for memory: {e}\nXML: {xml_response}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing memory XML: {e}\nXML: {xml_response}", exc_info=True)
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
        
        # Generate memory queries using LLM
        prompt = f"Create memory queries based on this conversation at location '{location}':\n\n{conversation_text}"
        logger.debug(f"Generating memory retrieval prompt for {self.agent_name}")
        queries_xml = self.llm_manager.generate_response(prompt, self.memory_retrieval_prompt)
        logger.debug(f"Received memory queries XML from LLM for {self.agent_name}: {queries_xml[:100]}...")

        # Parse the XML response to get queries
        queries = self._parse_memory_queries_xml(queries_xml)
        if not queries:
             logger.warning(f"Failed to parse memory queries XML for {self.agent_name}. XML: {queries_xml}")
             return [] # Return empty list if parsing failed

        # Execute each query and collect results
        all_memories = []
        for query in queries:
            memories = self._execute_memory_query(query)
            all_memories.extend(memories)
        
        # Remove duplicates (based on memory ID)
        unique_memories = []
        memory_ids = set()
        for memory in all_memories:
            if memory["id"] not in memory_ids:
                unique_memories.append(memory)
                memory_ids.add(memory["id"])
        
        # Sort by importance (descending) and return top results
        unique_memories.sort(key=lambda x: x.get("importance", 0), reverse=True)
        logger.info(f"Retrieved {len(unique_memories)} unique memories for {self.agent_name}.")
        return unique_memories[:9]  # Return up to 9 memories (3 per query)

    def _parse_memory_queries_xml(self, xml_response: str) -> List[Dict[str, Any]]:
        """
        Parse the XML response from the LLM into memory queries.
        
        Args:
            xml_response: XML-formatted memory queries from LLM
            
        Returns:
            List of query dictionaries
        """
        try:
            # Ensure the response is wrapped in a root element for valid XML parsing
            if not xml_response.strip().startswith('<'):
                 # Attempt to find the first < and last > if response has extra text
                 start = xml_response.find('<memory_queries>')
                 end = xml_response.rfind('</memory_queries>')
                 if start != -1 and end != -1:
                     xml_response = xml_response[start:end+len('</memory_queries>')]
                 else: # Give up if no root tag found
                     logger.warning(f"Invalid XML structure for query parsing: {xml_response}")
                     return []

            # Add a dummy root if necessary (ElementTree needs a single root)
            if not xml_response.strip().startswith('<root>'):
                 xml_response = f"<root>{xml_response}</root>"

            root = ET.fromstring(xml_response)
            queries = []
            for query_element in root.findall('.//query'):
                query = {}
                filters = {}
                for child in query_element:
                    if child.tag == "filters":
                        for filter_child in child:
                            field = filter_child.tag
                            value = filter_child.text.strip() if filter_child.text else ""
                            if field in ["min_importance", "max_importance"]:
                                try:
                                    filters[field] = int(value)
                                except ValueError:
                                    logger.warning(f"Invalid importance filter value '{value}' in query XML.")
                            else:
                                filters[field] = value
                    elif child.tag == "keywords":
                         query[child.tag] = [k.strip() for k in (child.text or "").split(",") if k.strip()]
                    else:
                        query[child.tag] = child.text.strip() if child.text else ""

                if filters:
                    query["filters"] = filters
                queries.append(query)

            logger.debug(f"Parsed {len(queries)} memory queries.")
            return queries[:3] # Limit to 3 queries

        except ET.ParseError as e:
            logger.error(f"XML parsing error for memory queries: {e}\nXML: {xml_response}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing memory queries XML: {e}\nXML: {xml_response}", exc_info=True)
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
        
        if "category" in filters:
            weaviate_filter["path"] = ["category"]
            weaviate_filter["operator"] = "Equal"
            weaviate_filter["valueString"] = filters["category"]
        
        importance_filters = []
        if "min_importance" in filters:
            importance_filters.append({
                "path": ["importance"],
                "operator": "GreaterThanEqual",
                "valueNumber": filters["min_importance"]
            })
        
        if "max_importance" in filters:
            importance_filters.append({
                "path": ["importance"],
                "operator": "LessThanEqual",
                "valueNumber": filters["max_importance"]
            })
        logger.debug(f"Executing memory query for {self.agent_name}: Type={search_type}, Filters={filters}")

        # Execute the appropriate search based on search type
        if search_type == "keyword":
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
