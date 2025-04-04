import os
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from ..llm.llm_manager import LLMManager

class MemoryUtils:
    """
    Utility functions for memory operations.
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """
        Initialize memory utilities.
        
        Args:
            llm_manager: Optional LLM manager for embeddings
        """
        self.llm_manager = llm_manager
    
    def set_llm_manager(self, llm_manager: LLMManager) -> None:
        """
        Set the LLM manager.
        
        Args:
            llm_manager: LLM manager instance
        """
        self.llm_manager = llm_manager
    
    def calculate_memory_importance(self, memory: Dict[str, Any]) -> int:
        """
        Calculate the importance of a memory.
        
        Args:
            memory: Memory object
            
        Returns:
            Importance score (1-10)
        """
        # If importance is already set, return it
        if "importance" in memory and isinstance(memory["importance"], (int, float)):
            return int(memory["importance"])
        
        # Default importance
        default_importance = 5
        
        # If we have an LLM manager, we could use it to calculate importance
        # based on the memory content, but for now we'll use a simple heuristic
        if "critical_information" in memory and memory["critical_information"]:
            # Longer critical information might indicate higher importance
            critical_info_length = len(memory["critical_information"])
            if critical_info_length > 200:
                return 8
            elif critical_info_length > 100:
                return 7
            elif critical_info_length > 50:
                return 6
        
        return default_importance
    
    def format_memory_for_prompt(self, memory: Dict[str, Any]) -> str:
        """
        Format a memory for inclusion in a prompt.
        
        Args:
            memory: Memory object
            
        Returns:
            Formatted memory string
        """
        formatted = f"Memory:\n"
        formatted += f"- Summary: {memory.get('summary', 'No summary')}\n"
        formatted += f"- Category: {memory.get('category', 'Uncategorized')}\n"
        
        # Format keywords as a comma-separated list
        keywords = memory.get('keywords', [])
        if isinstance(keywords, list):
            keywords_str = ", ".join(keywords)
        else:
            keywords_str = str(keywords)
        formatted += f"- Keywords: {keywords_str}\n"
        
        formatted += f"- Critical Information: {memory.get('critical_information', 'None')}\n"
        formatted += f"- Importance: {memory.get('importance', 5)}/10\n"
        formatted += f"- Time: {memory.get('timestamp', 'Unknown time')}\n"
        formatted += f"- Location: {memory.get('location', 'Unknown location')}\n"
        
        return formatted
    
    def merge_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple related memories into a single consolidated memory.
        
        Args:
            memories: List of memory objects to merge
            
        Returns:
            Merged memory object
        """
        if not memories:
            return {}
        
        if len(memories) == 1:
            return memories[0]
        
        # If we have an LLM manager, we could use it to generate a merged summary
        # For now, we'll use a simple approach
        
        # Collect all summaries
        summaries = [m.get("summary", "") for m in memories if "summary" in m]
        combined_summary = " ".join(summaries)
        
        # Take the most recent timestamp
        timestamps = [m.get("timestamp", "") for m in memories if "timestamp" in m]
        timestamps.sort(reverse=True)  # Sort in descending order
        latest_timestamp = timestamps[0] if timestamps else None
        
        # Collect all keywords
        all_keywords = []
        for memory in memories:
            if "keywords" in memory:
                if isinstance(memory["keywords"], list):
                    all_keywords.extend(memory["keywords"])
                else:
                    all_keywords.append(str(memory["keywords"]))
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in all_keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
        
        # Take the highest importance
        importances = [m.get("importance", 0) for m in memories if "importance" in m]
        max_importance = max(importances) if importances else 5
        
        # Combine critical information
        critical_infos = [m.get("critical_information", "") for m in memories if "critical_information" in m]
        combined_critical_info = " ".join(critical_infos)
        
        # Create the merged memory
        merged_memory = {
            "summary": f"Merged memory: {combined_summary[:200]}...",  # Truncate if too long
            "category": memories[0].get("category", "Uncategorized"),  # Use category from first memory
            "keywords": unique_keywords[:10],  # Limit to 10 keywords
            "critical_information": combined_critical_info,
            "importance": max_importance,
            "timestamp": latest_timestamp,
            "location": memories[0].get("location", "Unknown"),  # Use location from first memory
            "agent": memories[0].get("agent", "Unknown"),  # Use agent from first memory
            "id": f"merged_{memories[0].get('id', 'unknown')}"  # Create a new ID based on first memory
        }
        
        return merged_memory
