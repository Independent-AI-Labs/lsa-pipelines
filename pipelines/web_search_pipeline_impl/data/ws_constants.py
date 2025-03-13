"""
Constants, enums, and configuration values for the Web Search Assistant.

This module centralizes all constants, configuration values, and string literals
used throughout the assistant to make them easier to maintain and update.
"""
from enum import Enum

# System configuration
DEFAULT_MAX_WORKERS = 8
MAX_RECENT_MESSAGES = 10
MAX_TASK_HISTORY = 20
DEFAULT_INTENT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_SEARCH_TIMEOUT = 60  # Seconds
DEDICATED_CHAT_PREFIX = "🔍"
MAX_RESULTS_PER_PAGE = 15


# Knowledge Graph Node Types
class NodeType(str, Enum):
    """Enumeration of node types in the knowledge graph."""
    # Event Nodes (Read-only)
    USER_MESSAGE = "user_message"  # User message event
    SYSTEM_EVENT = "system_event"  # System-generated event

    # Data Nodes (Mutable)
    CHAT_MESSAGE = "chat_message"  # Chat message data (from user or assistant)
    SEARCH_CONFIG = "search_config"  # Search configuration
    SEARCH_RESULT = "search_result"  # Individual search result
    SEARCH_PATTERN = "search_pattern"  # Extracted pattern from search results
    CONTEXT = "context"  # Conversation context

    # Function Nodes
    WEB_SEARCH = "web_search"  # Web search function
    INTENT_ANALYSIS = "intent_analysis"  # Intent analysis function
    PATTERN_EXTRACTION = "pattern_extraction"  # Pattern extraction function

    # Outcome Nodes
    SEARCH_GOAL = "search_goal"  # Goal for a search operation
    INTENT_OUTCOME = "intent_outcome"  # Outcome of intent analysis


# Knowledge Graph Link Relations
class LinkRelation(str, Enum):
    """Enumeration of link relation types in the knowledge graph."""
    # Link relation types
    REPLY_TO = "reply_to"  # Message replies to another message
    PART_OF = "part_of"  # Component belongs to a larger entity
    DERIVES_FROM = "derives_from"  # Entity derived from another entity
    AUGMENTS = "augments"  # Entity augments/enhances another entity
    REFERENCES = "references"  # Entity references another entity
    CONTAINS = "contains"  # Entity contains another entity
    PRODUCED_BY = "produced_by"  # Entity was produced by a function/process
    TRIGGERS = "triggers"  # Event triggers a function/action
    HAS_OUTCOME = "has_outcome"  # Function/action has a specific outcome


# Intent Types
class IntentType(str, Enum):
    """Enumeration of user intent types."""
    SEARCH = "search"  # New search request
    FOLLOW_UP = "follow_up"  # Follow-up question on previous search
    AUGMENT_SEARCH = "augment_search"  # Augment existing search
    STATUS = "status"  # Check status of search
    CANCEL = "cancel"  # Cancel search
    HELP = "help"  # Request help
    CHAT = "chat"  # General chat/conversation


# Task Statuses
class TaskStatus(str, Enum):
    """Enumeration of task statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# System Operations
class Operation(str, Enum):
    """Enumeration of system operations."""
    CREATE_TASK = "create_task"
    CANCEL_TASK = "cancel_task"
    AUGMENT_TASK = "augment_task"
    SHOW_RESULTS = "show_results"
    SHOW_STATUS = "show_status"
    HELP = "help"
    DIRECT_REPLY = "direct_reply"


# Message templates
MSG_SEARCH_STARTED = "## 🔍 Started search for: {terms}"
MSG_SEARCH_COMPLETED = "## ✅ Search completed in {duration:.1f}s - Found {result_count} results and {pattern_count} patterns"
MSG_SEARCH_CANCELLED = "## 🛑 Search cancelled: {task_id}"
MSG_SEARCH_FAILED = "## ❌ Search failed: {error}"
MSG_SEARCH_AUGMENTED = "## 🔄 Search augmented with additional terms: {terms}"
MSG_NO_ACTIVE_TASKS = "## There are no active search tasks associated with this chat."
MSG_TASK_NOT_FOUND = "## No task found with ID: {task_id}"
MSG_INVALID_COMMAND = "## I didn't understand that command. Type 'help' for assistance."

# UI messages
UI_THINKING = "<think>Analyzing your request...</think>"
UI_DEDICATED_CHAT_TITLE = "🔍 {search_terms}"

# Error messages
ERR_TASK_NOT_FOUND = "Task not found: {task_id}"
ERR_INVALID_TASK_ID = "Invalid task ID: {task_id}"
ERR_TASK_ALREADY_COMPLETED = "Task already completed: {task_id}"
ERR_TASK_ALREADY_CANCELLED = "Task already cancelled: {task_id}"
ERR_SEARCH_FAILED = "Search failed: {error}"
ERR_GRAPH_ERROR = "Knowledge graph error: {error}"
ERR_NO_CONTEXT = "No context available for chat: {chat_id}"

# KG serialization constants
MAX_FIELD_LENGTH = 128  # Maximum length of serialized Node/Link fields before truncation
MAX_KG_NODES = 50  # Maximum number of nodes to include in a serialized KG subset
