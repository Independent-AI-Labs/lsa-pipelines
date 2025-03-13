"""
Knowledge Graph Manager for the Web Search Assistant.

This module manages the integration with the Knowledge Graph framework,
handling the creation, updating, and querying of nodes and relationships.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from integration.data.config import PYTHON_INSTR_MODEL
from integration.net.ollama.ollama_api import prompt_model
from integration.net.www.chrome.chrome_surfer import search_web
from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants import (
    MAX_RECENT_MESSAGES, NodeType, TaskStatus, LinkRelation
)
from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_models import (
    ChatMessage, ConversationContext, SearchConfig, SearchResult, SearchPattern
)
from knowledge.graph.kg_models import KnowledgeGraph


class KnowledgeGraphManager:
    """
    Manages the creation, updating, and querying of nodes and relationships
    in the Knowledge Graph for the Web Search Assistant.
    """

    def __init__(self):
        """Initialize the Knowledge Graph Manager."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.graph = KnowledgeGraph()
        self._initialize_hooks()
        self._executor = None

    def _initialize_hooks(self):
        """Initialize event hooks for the knowledge graph."""
        self.graph.register_hook("node_added", self._log_node_event)
        self.graph.register_hook("node_updated", self._log_node_event)
        self.graph.register_hook("edge_added", self._log_edge_event)

    def _log_node_event(self, event: str, data: Dict[str, Any]):
        """Log node events for debugging."""
        node_type = data.get("type", "unknown")
        node_id = data.get("uid", "unknown")
        self.logger.debug(f"{event}: {node_type} - {node_id}")

    def _log_edge_event(self, event: str, data: Dict[str, Any]):
        """Log edge events for debugging."""
        relation = data.get("relation", "unknown")
        source = data.get("source_uid", "unknown")
        target = data.get("target_uid", "unknown")
        self.logger.debug(f"{event}: {relation} - {source} -> {target}")

    async def initialize(self):
        """Initialize resources asynchronously."""
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=8)
        self.logger.info("KnowledgeGraphManager initialized")

    async def shutdown(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
        self.logger.info("KnowledgeGraphManager shut down")

    def get_or_create_context(self, chat_id: str) -> ConversationContext:
        """
        Get or create a conversation context for a chat.

        Args:
            chat_id: The chat ID

        Returns:
            The ConversationContext
        """
        try:
            # Try to get existing context
            context_nodes = self.graph.query_nodes(type=NodeType.CONTEXT, chat_id=chat_id)
            if context_nodes:
                # Convert the KGNode to a ConversationContext object
                node = context_nodes[0]
                node_data = node.model_dump()

                # Create a new ConversationContext with the complete data
                context = ConversationContext(**node_data)
                return context

            # Create new context
            with self.graph.transaction():
                context_id = str(uuid.uuid4())
                context = ConversationContext(
                    context_id=context_id,
                    chat_id=chat_id,
                    active_configs=[],
                    recent_messages=[],
                    topics=[],
                    entities={}
                )
                self.graph.add_node(context)
                return context
        except Exception as e:
            self.logger.error(f"Error getting/creating context: {str(e)}")
            # Create a minimal valid context even if the operation fails
            context_id = str(uuid.uuid4())
            return ConversationContext(
                context_id=context_id,
                chat_id=chat_id,
                active_configs=[],
                recent_messages=[],
                topics=[],
                entities={}
            )

    def get_recent_messages(self, chat_id: str, role: Optional[str] = None,
                            limit: int = MAX_RECENT_MESSAGES) -> List[ChatMessage]:
        """
        Get recent messages for a chat.

        Args:
            chat_id: The chat ID
            role: Optional role to filter by (user/assistant)
            limit: Maximum number of messages to return

        Returns:
            List of ChatMessage objects
        """
        query_params = {"chat_id": chat_id, "type": NodeType.CHAT_MESSAGE}
        if role:
            query_params["role"] = role

        nodes = self.graph.query_nodes(**query_params)

        # Sort by timestamp
        nodes.sort(key=lambda x: x.timestamp if hasattr(x, 'timestamp') else 0, reverse=True)

        # Limit the number of results
        return nodes[:limit]

    def get_search_config(self, config_id: str) -> Optional[SearchConfig]:
        """
        Get a search configuration by its ID.

        Args:
            config_id: The config ID

        Returns:
            The SearchConfig if found, None otherwise
        """
        try:
            nodes = self.graph.query_nodes(type=NodeType.SEARCH_CONFIG, config_id=config_id)
            if not nodes:
                # Try by UID if not found by config_id
                try:
                    node = self.graph.get_node(f"config-{config_id}")
                    if node and node.type == NodeType.SEARCH_CONFIG:
                        return node
                except:
                    pass
                return None

            return nodes[0]
        except Exception as e:
            self.logger.error(f"Error getting search config: {str(e)}")
            return None

    def get_recent_completed_configs(self, chat_id: str, limit: int = 3) -> List[SearchConfig]:
        """
        Get recent completed search configurations for a chat.

        Args:
            chat_id: The chat ID
            limit: Maximum number of configs to return

        Returns:
            List of completed SearchConfig objects
        """
        try:
            nodes = self.graph.query_nodes(
                type=NodeType.SEARCH_CONFIG,
                chat_id=chat_id,
                status=TaskStatus.COMPLETED
            )

            # Sort by end_time (most recent first)
            nodes.sort(
                key=lambda x: x.end_time if hasattr(x, 'end_time') and x.end_time is not None else 0,
                reverse=True
            )

            return nodes[:limit]
        except Exception as e:
            self.logger.error(f"Error getting completed search configs: {str(e)}")
            return []

    def get_active_configs(self, chat_id: str) -> List[SearchConfig]:
        """
        Get active (pending or running) search configurations for a chat.

        Args:
            chat_id: The chat ID

        Returns:
            List of active SearchConfig objects
        """
        try:
            pending_nodes = self.graph.query_nodes(
                type=NodeType.SEARCH_CONFIG,
                chat_id=chat_id,
                status=TaskStatus.PENDING
            )

            running_nodes = self.graph.query_nodes(
                type=NodeType.SEARCH_CONFIG,
                chat_id=chat_id,
                status=TaskStatus.RUNNING
            )

            # Combine and sort by start_time
            active_nodes = pending_nodes + running_nodes
            active_nodes.sort(
                key=lambda x: x.start_time if hasattr(x, 'start_time') and x.start_time is not None else 0,
                reverse=True
            )

            return active_nodes
        except Exception as e:
            self.logger.error(f"Error getting active search configs: {str(e)}")
            return []

    def get_results_for_config(self, config_id: str) -> List[SearchResult]:
        """
        Get search results for a specific configuration.

        Args:
            config_id: The config ID

        Returns:
            List of SearchResult objects
        """
        try:
            nodes = self.graph.query_nodes(type=NodeType.SEARCH_RESULT, config_id=config_id)

            # Sort by relevance score if available, otherwise just return in any order
            nodes.sort(
                key=lambda x: x.relevance_score if hasattr(x, 'relevance_score') and x.relevance_score is not None else 0,
                reverse=True
            )

            return nodes
        except Exception as e:
            self.logger.error(f"Error getting results for config: {str(e)}")
            return []

    def get_patterns_for_config(self, config_id: str) -> List[SearchPattern]:
        """
        Get search patterns for a specific configuration.

        Args:
            config_id: The config ID

        Returns:
            List of SearchPattern objects
        """
        try:
            nodes = self.graph.query_nodes(type=NodeType.SEARCH_PATTERN, config_id=config_id)

            # Sort by confidence if available
            nodes.sort(
                key=lambda x: x.confidence if hasattr(x, 'confidence') and x.confidence is not None else 0,
                reverse=True
            )

            return nodes
        except Exception as e:
            self.logger.error(f"Error getting patterns for config: {str(e)}")
            return []

    def execute_search(self, config: SearchConfig) -> Tuple[List[str], List[str]]:
        """
        Execute a web search based on a search configuration.

        Args:
            config: The search configuration

        Returns:
            Tuple of (results, patterns) lists
        """
        try:
            # Use chrome_surfer to execute the search
            results, patterns = search_web(
                search_terms=config.search_terms,
                semantic_patterns=config.semantic_patterns,
                instructions=config.instructions,
                max_results=config.max_results
            )

            return results, patterns
        except Exception as e:
            self.logger.error(f"Error executing search: {str(e)}")
            return [], []

    def extract_patterns(self, results: List[str], config_id: Optional[str] = None) -> List[str]:
        """
        Extract patterns from search results.

        Args:
            results: The search results to analyze
            config_id: Optional config ID for context

        Returns:
            List of extracted patterns
        """
        try:
            # If no results, return empty list
            if not results:
                return []

            # Get search config for context if available
            config = None
            search_terms = []
            if config_id:
                config = self.get_search_config(config_id)
                if config:
                    search_terms = config.search_terms

            # Prepare the prompt for pattern extraction
            prompt = f"""
            Analyze the following search results and identify 3-5 key patterns, insights, or themes.
            Focus on extracting meaningful observations that would help a user understand the topic better.
            
            Search Topic: {', '.join(search_terms) if search_terms else "Unknown"}
            
            Search Results:
            """

            # Add results to the prompt (limit to prevent excessive token usage)
            for i, result in enumerate(results[:15], 1):
                prompt += f"\n{i}. {result}"

            # Call the LLM to extract patterns
            response = prompt_model(
                message=prompt,
                model=PYTHON_INSTR_MODEL,
                system_prompt="You are a pattern extraction assistant that analyzes search results to identify key insights and themes. Respond with a numbered list of patterns, one per line."
            )

            # Parse the response into individual patterns
            patterns = []
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Look for numbered lines or bullet points
                if (line and (line[0].isdigit() and line[1:3] in ['. ', ') ']) or
                        line.startswith('- ') or line.startswith('• ')):
                    # Remove the numbering/bullet and add to patterns
                    pattern_text = line[line.find(' ') + 1:].strip()
                    if pattern_text:
                        patterns.append(pattern_text)

            # If no patterns were extracted with the above method, try to use the whole response
            if not patterns and response.strip():
                # Split by sentences or paragraphs
                import re
                potential_patterns = re.split(r'(?<=[.!?])\s+', response.strip())
                # Take up to 5 non-empty items
                patterns = [p.strip() for p in potential_patterns if p.strip()][:5]

            return patterns
        except Exception as e:
            self.logger.error(f"Error extracting patterns: {str(e)}")
            return []

    def cancel_config(self, config_id: str) -> bool:
        """
        Cancel a search configuration.

        Args:
            config_id: The config ID

        Returns:
            True if the config was cancelled, False otherwise
        """
        try:
            with self.graph.transaction():
                config = self.get_search_config(config_id)
                if not config:
                    return False

                # Only cancel if not already completed or cancelled
                if config.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    config.status = TaskStatus.CANCELLED
                    config.end_time = datetime.now().timestamp()
                    self.graph.update_node(config.uid, config.model_dump())
                    return True

                return False
        except Exception as e:
            self.logger.error(f"Error cancelling config: {str(e)}")
            return False

    def cancel_all_configs(self, chat_id: str) -> int:
        """
        Cancel all active search configurations for a chat.

        Args:
            chat_id: The chat ID

        Returns:
            Number of configs cancelled
        """
        try:
            with self.graph.transaction():
                active_configs = self.get_active_configs(chat_id)
                count = 0

                for config in active_configs:
                    config.status = TaskStatus.CANCELLED
                    config.end_time = datetime.now().timestamp()
                    self.graph.update_node(config.uid, config.model_dump())
                    count += 1

                return count
        except Exception as e:
            self.logger.error(f"Error cancelling all configs: {str(e)}")
            return 0

    def get_config_results_dict(self, config_id: str) -> Dict[str, Any]:
        """
        Get a dictionary with config details and results.

        Args:
            config_id: The config ID

        Returns:
            Dictionary with config details and results
        """
        config = self.get_search_config(config_id)
        if not config:
            return {}

        # Get results and patterns as lists of strings
        results = [result.content for result in self.get_results_for_config(config_id)]
        patterns = [pattern.content for pattern in self.get_patterns_for_config(config_id)]

        # Find related configs (those that augment this one or are augmented by this one)
        related_configs = []

        # Find augmenting relationships (B augments A)
        augmenting_links = self.graph.query_edges(
            relation=LinkRelation.AUGMENTS,
            target_uid=config.uid
        )
        for link in augmenting_links:
            source_config = self.get_search_config(link.source_uid.split('-')[-1])
            if source_config:
                related_configs.append(source_config.config_id)

        # Find augmented relationships (A augments B)
        augmented_links = self.graph.query_edges(
            relation=LinkRelation.AUGMENTS,
            source_uid=config.uid
        )
        for link in augmented_links:
            target_config = self.get_search_config(link.target_uid.split('-')[-1])
            if target_config:
                related_configs.append(target_config.config_id)

        return {
            "config_id": config.config_id,
            "search_terms": config.search_terms,
            "status": config.status,
            "duration": config.duration(),
            "results": results,
            "patterns": patterns,
            "related_configs": related_configs
        }

    def snapshot(self) -> int:
        """
        Create a snapshot of the current knowledge graph state.

        Returns:
            The snapshot version number
        """
        return self.graph.snapshot(message="Automatic snapshot by KnowledgeGraphManager")
