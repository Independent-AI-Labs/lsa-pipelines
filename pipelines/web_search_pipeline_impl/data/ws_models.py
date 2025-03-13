"""
Web Search Assistant models that integrate with the Knowledge Graph and
Recursive Reasoning Framework.

These models extend the base Knowledge Graph classes and provide specialized
functionality for web search operations while maintaining proper abstraction.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import Field

from integration.data.config import TOP_N_WEB_SEARCH_RESULTS
from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants import (
    NodeType, LinkRelation, IntentType, TaskStatus, MAX_RECENT_MESSAGES
)
from knowledge.reasoning.engine.re_models import (
    DataNode, FunctionNode, OutcomeNode, Link as ReasoningLink, ActionMetaNode, EventMetaNode
)


# -----------------------------------------------------------------------------
# Event Meta-Nodes: Read-Only Records of Events
# -----------------------------------------------------------------------------

class UserMessageEvent(EventMetaNode):
    """
    Read-only record of a user message event.

    Captures metadata about a user message that triggered system actions.
    """
    chat_id: str
    user_id: Optional[str] = None
    message_id: str
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"event-message-{data.get('message_id', str(uuid.uuid4()))}"
        # Set the specific node type
        data["type"] = NodeType.USER_MESSAGE
        super().__init__(**data)


class SystemEvent(EventMetaNode):
    """
    Read-only record of a system-generated event.

    Captures metadata about internal system events that trigger actions.
    """
    event_type: str
    source: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"event-system-{data.get('event_id', str(uuid.uuid4()))}"
        # Set the specific node type
        data["type"] = NodeType.SYSTEM_EVENT
        super().__init__(**data)


# -----------------------------------------------------------------------------
# Action Meta-Nodes: Read-Only Records of Actions
# -----------------------------------------------------------------------------

class IntentAnalysisAction(ActionMetaNode):
    """
    Read-only record of an intent analysis action.

    Represents a decision or reasoning step taken by the system to analyze
    user intent from a message.
    """
    message_id: str
    chat_id: str
    intent_type: IntentType
    confidence: float = 1.0
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"action-intent-{data.get('message_id', str(uuid.uuid4()))}"
        super().__init__(**data)


# -----------------------------------------------------------------------------
# Data Nodes: Mutable Content Nodes
# -----------------------------------------------------------------------------

class ChatMessage(DataNode):
    """
    Represents a message in a conversation (either from user or assistant).

    Stores the actual message content and metadata.
    """
    message_id: str
    chat_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"message-{data.get('message_id', str(uuid.uuid4()))}"
        # Set the specific node type
        data["type"] = NodeType.CHAT_MESSAGE
        super().__init__(**data)


class SearchConfig(DataNode):
    """
    Configuration parameters for a web search operation.

    Defines what to search for, how to search, and any constraints or preferences.
    """
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chat_id: str
    search_terms: List[str]
    semantic_patterns: Optional[List[str]] = None
    instructions: Optional[str] = None
    max_results: int = TOP_N_WEB_SEARCH_RESULTS
    status: TaskStatus = TaskStatus.PENDING
    parent_config_id: Optional[str] = None  # For derived/augmented searches
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"config-{data.get('config_id', str(uuid.uuid4()))}"
        # Set the specific node type
        data["type"] = NodeType.SEARCH_CONFIG
        super().__init__(**data)

    def duration(self) -> float:
        """Calculate the duration of the search in seconds"""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else datetime.now().timestamp()
        return end - self.start_time


class SearchResult(DataNode):
    """
    Individual result from a web search.

    Contains the content and metadata for a single search result.
    """
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config_id: str  # The search config this result is associated with
    content: str
    source: Optional[str] = None
    relevance_score: Optional[float] = None
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"result-{data.get('result_id', str(uuid.uuid4()))}"
        # Set the specific node type
        data["type"] = NodeType.SEARCH_RESULT
        super().__init__(**data)


class SearchPattern(DataNode):
    """
    Pattern or insight extracted from search results.

    Represents a detected pattern, trend, or insight across multiple search results.
    """
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config_id: str  # The search config this pattern is associated with
    content: str
    confidence: Optional[float] = None
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"pattern-{data.get('pattern_id', str(uuid.uuid4()))}"
        # Set the specific node type
        data["type"] = NodeType.SEARCH_PATTERN
        super().__init__(**data)


class ConversationContext(DataNode):
    """
    Context information for a conversation.

    Tracks conversation state, active searches, topics, and entities to improve
    intent detection and response generation.
    """
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chat_id: str
    active_configs: List[str] = Field(default_factory=list)  # IDs of active search configs
    recent_messages: List[str] = Field(default_factory=list)  # IDs of recent messages
    topics: List[str] = Field(default_factory=list)  # Current conversation topics
    entities: Dict[str, Any] = Field(default_factory=dict)  # Entities extracted from conversation
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"context-{data.get('context_id', str(uuid.uuid4()))}"
        # Set the specific node type
        data["type"] = NodeType.CONTEXT
        super().__init__(**data)

    def update_with_message(self, message_node: ChatMessage) -> 'ConversationContext':
        """Update context with a new message"""
        if message_node.message_id not in self.recent_messages:
            self.recent_messages.append(message_node.message_id)
            # Keep only the most recent messages (limit from constants)
            if len(self.recent_messages) > MAX_RECENT_MESSAGES:
                self.recent_messages.pop(0)
        return self

    def update_with_search_config(self, config: SearchConfig) -> 'ConversationContext':
        """Update context with a new search configuration"""
        # Add to active configs if pending or running
        if config.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if config.config_id not in self.active_configs:
                self.active_configs.append(config.config_id)

        # Update entities with search terms
        for term in config.search_terms:
            self.entities[term] = {
                "type": "search_term",
                "config_id": config.config_id
            }

            # Simple entity extraction (could be improved with NLP)
            words = term.split()
            if len(words) >= 2:
                for word in words:
                    if word[0].isupper() and word.lower() not in ["what", "where", "when", "who", "how"]:
                        self.entities[word] = {"type": "entity", "related_term": term}

        return self


# -----------------------------------------------------------------------------
# Function Nodes: Executable Operations
# -----------------------------------------------------------------------------

class WebSearchFunction(FunctionNode):
    """
    System function that performs web searches.

    This predefined FunctionNode takes a SearchConfig and produces SearchResults
    when evaluated.
    """

    def evaluate(self, instructions: Any = None) -> Any:
        """
        Evaluate this function, performing a web search based on the provided
        search configuration or instructions.

        Args:
            instructions: Either a SearchConfig node, a config_id string, or natural
                         language instructions for creating a new search

        Returns:
            A dictionary containing the search results and metadata
        """
        # The actual implementation will be injected at runtime
        # This is just the interface definition
        return super().evaluate(instructions)


class PatternExtractionFunction(FunctionNode):
    """
    System function that extracts patterns from search results.

    This predefined FunctionNode analyzes a set of SearchResults and identifies
    patterns, trends, or insights.
    """

    def evaluate(self, instructions: Any = None) -> Any:
        """
        Evaluate this function, extracting patterns from the provided search results.

        Args:
            instructions: Either a list of SearchResult nodes, a config_id to find
                         results for, or natural language instructions

        Returns:
            A list of extracted patterns as SearchPattern objects
        """
        # The actual implementation will be injected at runtime
        # This is just the interface definition
        return super().evaluate(instructions)


# -----------------------------------------------------------------------------
# Outcome Nodes: Goal and Evaluation Nodes
# -----------------------------------------------------------------------------

class SearchGoal(OutcomeNode):
    """
    Represents the goal or objective of a search operation.

    Used to track whether a search has successfully achieved its intended purpose.
    """
    config_id: str
    goal_description: str
    target_eval_state: Any = True  # Default target is successful completion
    verification_criteria: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"goal-{data.get('config_id', str(uuid.uuid4()))}"
        # Set the specific node type
        data["type"] = NodeType.SEARCH_GOAL
        super().__init__(**data)


class IntentOutcome(OutcomeNode):
    """
    Represents the outcome of an intent analysis.

    Tracks whether the system correctly identified and acted on user intent.
    """
    message_id: str
    intent_type: IntentType
    target_eval_state: Any = True  # Default target is successful handling

    def __init__(self, **data):
        if "uid" not in data:
            data["uid"] = f"outcome-intent-{data.get('message_id', str(uuid.uuid4()))}"
        # Set the specific node type
        data["type"] = NodeType.INTENT_OUTCOME
        super().__init__(**data)


# -----------------------------------------------------------------------------
# Link (Edge) Definitions
# -----------------------------------------------------------------------------

class WebSearchLink(ReasoningLink):
    """
    Represents relationships between nodes in the web search knowledge graph.

    Extends the base ReasoningLink with web search-specific relation types.
    """
    relation: LinkRelation

    def __init__(self, **data):
        # Ensure we have a source and target UID
        if "source_uid" not in data or "target_uid" not in data:
            raise ValueError("Links must have both source_uid and target_uid")

        # Create a unique ID if not provided
        if "uid" not in data:
            source = data.get("source_uid", "").split("-")[-1]
            target = data.get("target_uid", "").split("-")[-1]
            data["uid"] = f"link-{source}-{target}"

        super().__init__(**data)
