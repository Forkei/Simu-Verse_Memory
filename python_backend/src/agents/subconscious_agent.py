import datetime
import uuid
from typing import Dict, List, Any, Optional
from ..llm.llm_manager import LLMManager
from ..memory.weaviate_client import WeaviateClient

class SubconsciousAgent:
    """
    Subconscious agent that manages memories for a main agent.
    """
    
    def __init__(self, agent_name: str, llm_manager: LLMManager, 
                 weaviate_client: WeaviateClient, memory_categories: Dict[str, Any]):
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
        
        # System prompt for memory retrieval
        self.memory_retrieval_prompt = self._get_memory_retrieval_prompt()
    
    def _get_memory_creation_prompt(self) -> str:
        """Get the system prompt for memory creation."""
        categories = ", ".join([cat["name"] for cat in self.memory_categories["categories"]])
        
        prompt = f"""You are the subconscious mind of {self.agent_name}. Your task is to create a memory from recent conversation.
        
Analyze the conversation and create a memory with the following components:
1. Summary: A concise description of what happened
2. Category: Choose from: {categories}
3. Keywords: 3-5 relevant keywords for future retrieval
4. Critical Information: Details that will help make good decisions when this memory is recalled
5. Importance: Rate from 1-10 how important this memory is (1 = trivial, 10 = life-changing)

Respond in XML format like this:
<memory>
  <summary>Brief summary of what happened</summary>
  <category>one_of_the_categories</category>
  <keywords>keyword1, keyword2, keyword3</keywords>
  <critical_information>Important details that should influence future decisions</critical_information>
  <importance>7</importance>
</memory>
"""
        return prompt
    
    def _get_memory_retrieval_prompt(self) -> str:
        """Get the system prompt for memory retrieval."""
        prompt = f"""You are the subconscious mind of {self.agent_name}. Your task is to create queries to retrieve relevant memories.

Based on the recent conversation and current context, create up to 3 memory queries that will help retrieve the most relevant memories.

For each query, specify:
1. Search type: "keyword", "semantic", or "hybrid"
2. Keywords: If using keyword or hybrid search
3. Query text: For semantic or hybrid search
4. Filters: Any filters to apply (category, min/max importance, time range)

Respond in XML format like this:
<memory_queries>
  <query>
    <search_type>hybrid</search_type>
    <keywords>meeting, project, deadline</keywords>
    <query_text>Important information about the project deadline</query_text>
    <filters>
      <category>conversation</category>
      <min_importance>5</min_importance>
    </filters>
  </query>
  <!-- Additional queries as needed, up to 3 total -->
</memory_queries>
"""
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
        memory_xml = self.llm_manager.generate_response(prompt, self.memory_creation_prompt)
        
        # Parse the XML response
        memory = self._parse_memory_xml(memory_xml)
        
        # Add timestamp and location
        memory["timestamp"] = datetime.datetime.now().isoformat()
        memory["location"] = location
        memory["agent"] = self.agent_name
        memory["id"] = str(uuid.uuid4())
        
        # Store in Weaviate
        self.weaviate_client.add_object(
            collection_name=self.collection_name,
            properties=memory,
            vector_field="summary"  # Use summary for vector embedding
        )
        
        return memory
    
    def _parse_memory_xml(self, xml_response: str) -> Dict[str, Any]:
        """
        Parse the XML response from the LLM into a memory object.
        
        Args:
            xml_response: XML-formatted memory from LLM
            
        Returns:
            Dictionary representation of the memory
        """
        # This is a simplified parser - in a real implementation, use a proper XML parser
        memory = {}
        
        for field in ["summary", "category", "keywords", "critical_information", "importance"]:
            start_tag = f"<{field}>"
            end_tag = f"</{field}>"
            
            if start_tag in xml_response and end_tag in xml_response:
                start_idx = xml_response.find(start_tag) + len(start_tag)
                end_idx = xml_response.find(end_tag)
                value = xml_response[start_idx:end_idx].strip()
                
                # Convert importance to integer
                if field == "importance":
                    try:
                        value = int(value)
                    except ValueError:
                        value = 5  # Default importance
                
                memory[field] = value
        
        # Convert keywords from comma-separated string to list
        if "keywords" in memory:
            memory["keywords"] = [k.strip() for k in memory["keywords"].split(",")]
        
        return memory
    
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
        queries_xml = self.llm_manager.generate_response(prompt, self.memory_retrieval_prompt)
        
        # Parse the XML response to get queries
        queries = self._parse_memory_queries_xml(queries_xml)
        
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
        return unique_memories[:9]  # Return up to 9 memories (3 per query)
    
    def _parse_memory_queries_xml(self, xml_response: str) -> List[Dict[str, Any]]:
        """
        Parse the XML response from the LLM into memory queries.
        
        Args:
            xml_response: XML-formatted memory queries from LLM
            
        Returns:
            List of query dictionaries
        """
        # This is a simplified parser - in a real implementation, use a proper XML parser
        queries = []
        
        # Split into individual queries
        if "<query>" in xml_response and "</query>" in xml_response:
            query_sections = xml_response.split("<query>")[1:]
            
            for section in query_sections:
                if "</query>" in section:
                    query_text = section.split("</query>")[0].strip()
                    query = {}
                    
                    # Extract search type
                    if "<search_type>" in query_text and "</search_type>" in query_text:
                        start_idx = query_text.find("<search_type>") + len("<search_type>")
                        end_idx = query_text.find("</search_type>")
                        query["search_type"] = query_text[start_idx:end_idx].strip()
                    
                    # Extract keywords
                    if "<keywords>" in query_text and "</keywords>" in query_text:
                        start_idx = query_text.find("<keywords>") + len("<keywords>")
                        end_idx = query_text.find("</keywords>")
                        keywords_str = query_text[start_idx:end_idx].strip()
                        query["keywords"] = [k.strip() for k in keywords_str.split(",")]
                    
                    # Extract query text
                    if "<query_text>" in query_text and "</query_text>" in query_text:
                        start_idx = query_text.find("<query_text>") + len("<query_text>")
                        end_idx = query_text.find("</query_text>")
                        query["query_text"] = query_text[start_idx:end_idx].strip()
                    
                    # Extract filters
                    query["filters"] = {}
                    if "<filters>" in query_text and "</filters>" in query_text:
                        filters_text = query_text.split("<filters>")[1].split("</filters>")[0].strip()
                        
                        # Category filter
                        if "<category>" in filters_text and "</category>" in filters_text:
                            start_idx = filters_text.find("<category>") + len("<category>")
                            end_idx = filters_text.find("</category>")
                            query["filters"]["category"] = filters_text[start_idx:end_idx].strip()
                        
                        # Min importance filter
                        if "<min_importance>" in filters_text and "</min_importance>" in filters_text:
                            start_idx = filters_text.find("<min_importance>") + len("<min_importance>")
                            end_idx = filters_text.find("</min_importance>")
                            try:
                                query["filters"]["min_importance"] = int(filters_text[start_idx:end_idx].strip())
                            except ValueError:
                                pass
                        
                        # Max importance filter
                        if "<max_importance>" in filters_text and "</max_importance>" in filters_text:
                            start_idx = filters_text.find("<max_importance>") + len("<max_importance>")
                            end_idx = filters_text.find("</max_importance>")
                            try:
                                query["filters"]["max_importance"] = int(filters_text[start_idx:end_idx].strip())
                            except ValueError:
                                pass
                    
                    queries.append(query)
        
        return queries[:3]  # Limit to 3 queries
    
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
        
        # Execute the appropriate search based on search type
        if search_type == "keyword":
            # Keyword search
            keywords = query.get("keywords", [])
            if not keywords:
                return []
            
            # Join keywords with OR for broader search
            keyword_query = " OR ".join(keywords)
            
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
                return []
            
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
                
                return self.weaviate_client.hybrid_search(
                    collection_name=self.collection_name,
                    query=query_text,
                    keyword_query=keyword_query,
                    filters=weaviate_filter,
                    limit=3
                )
            elif query_text:
                # If we only have query text, do semantic search
                return self.weaviate_client.semantic_search(
                    collection_name=self.collection_name,
                    query=query_text,
                    filters=weaviate_filter,
                    limit=3
                )
            else:
                # If we only have keywords, do keyword search
                keyword_query = " OR ".join(keywords)
                
                return self.weaviate_client.keyword_search(
                    collection_name=self.collection_name,
                    query=keyword_query,
                    filters=weaviate_filter,
                    limit=3
                )
