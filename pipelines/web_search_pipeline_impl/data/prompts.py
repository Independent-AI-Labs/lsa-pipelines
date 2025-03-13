"""
Dynamic prompt generation for the Web Search Assistant.

This module provides functions that generate specialized prompts for different
operations including KGML processing, instruction parsing, and more.
Each function adapts its output based on the specific context and requirements.
"""

from typing import Dict, List, Optional, Any

from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants import (
    NodeType
)


def generate_instruction_parser_prompt(entity_type: str, command_type: str,
                                       node_type: Optional[str] = None) -> str:
    """
    Generate a system prompt for instruction parsing based on operation context.

    Creates a specialized system prompt that guides the model in generating
    Python code for a specific command and entity type.

    Args:
        entity_type: Entity type (NODE or LINK)
        command_type: Command type (C, U, D, or E)
        node_type: Optional specific node type (e.g., NodeType.SEARCH_CONFIG)

    Returns:
        A system prompt for instruction parsing
    """
    base_prompt = """
You are a specialized code generator that converts natural language instructions 
to secure Python code for knowledge graph operations.

Your generated code MUST:
1. Only manipulate the provided object ('node' or 'edge')
2. Use only basic Python operations (no imports except those explicitly allowed)
3. Store results in the 'result' variable as a dictionary
4. Be secure and not attempt to access any system resources
5. Not use exec(), eval(), or any other dangerous functions
6. Not import any modules except: math, datetime, json, re
7. Not try to access filesystem, network, or environment variables
8. Focus ONLY on the specific operation requested

RETURN ONLY THE PYTHON CODE - NO EXPLANATIONS OR MARKDOWN.
    """

    # Add specific guidance based on entity type and command
    if entity_type == "NODE":
        # Get specific guidance for node type if provided
        node_type_guidance = ""
        if node_type:
            node_type_guidance = _get_node_type_specific_guidance(node_type, command_type)

        if command_type == "C":
            # Create node command
            return base_prompt + f"""
The 'node' variable is a KGNode object you need to populate based on the instruction.
You can set:
- node.type: str - The node's type
- node.meta_props: Dict[str, Any] - Dictionary of metadata properties

For timestamp fields, use:
from datetime import datetime
timestamp = datetime.now().isoformat()

{node_type_guidance}

Remember to populate the 'result' variable with a dictionary containing at least:
{{'status': 'created', 'node_id': node.uid, 'node_type': node.type}}
"""
        elif command_type == "U":
            # Update node command
            return base_prompt + f"""
The 'node' variable is an existing KGNode object you need to update based on the instruction.
You can modify:
- node.meta_props: Dict[str, Any] - Dictionary of metadata properties
- Any other specific properties the node type might have

DO NOT modify the node's uid or type.

{node_type_guidance}

Update timestamp:
from datetime import datetime
node.meta_props['updated_at'] = datetime.now().isoformat()

Remember to populate the 'result' variable with a dictionary containing at least:
{{'status': 'updated', 'node_id': node.uid}}
"""
        elif command_type == "E":
            # Evaluate node command
            return base_prompt + f"""
The 'node' variable is an existing KGNode object you need to evaluate.
You can access:
- node.uid: The node's unique identifier
- node.type: The node's type
- node.meta_props: Dictionary of metadata properties
- Any other specific properties the node type might have

{node_type_guidance}

Your code should compute a result based on the node's data and the instruction.
Store your evaluation results in the 'result' variable.

For function nodes, 'result' should include:
{{'status': 'evaluating', 'function': node.type, 'parameters': <extracted parameters>}}
"""
        elif command_type == "D":
            # Delete node command
            return base_prompt + """
The 'node' variable is an existing KGNode object that will be deleted.
You should perform any necessary cleanup or validation before deletion.

Remember to populate the 'result' variable with a dictionary containing at least:
{'status': 'deleting', 'node_id': node.uid}
"""

    elif entity_type == "LINK":
        if command_type == "C":
            # Create link command
            return base_prompt + """
The 'edge' variable is a KGEdge object you need to populate based on the instruction.
You must set:
- edge.source_uid: str - The source node's uid
- edge.target_uid: str - The target node's uid
- edge.relation: LinkRelation - The relation type (use a value from the LinkRelation enum)

Set timestamp:
from datetime import datetime
edge.meta_props['created_at'] = datetime.now().isoformat()

Valid relation types from LinkRelation enum:
REPLY_TO, PART_OF, DERIVES_FROM, AUGMENTS, REFERENCES, PRODUCES, TRIGGERS, HAS_OUTCOME

Remember to populate the 'result' variable with a dictionary containing at least:
{'status': 'created', 'link_id': edge.uid, 'relation': edge.relation}
"""
        elif command_type == "U":
            # Update link command
            return base_prompt + """
The 'edge' variable is an existing KGEdge object you need to update based on the instruction.
You can modify:
- edge.relation: LinkRelation - The relation type (use a value from the LinkRelation enum)
- edge.meta_props: Dict[str, Any] - Dictionary of metadata properties

DO NOT modify the edge's source_uid or target_uid.

Update timestamp:
from datetime import datetime
edge.meta_props['updated_at'] = datetime.now().isoformat()

Remember to populate the 'result' variable with a dictionary containing at least:
{'status': 'updated', 'link_id': edge.uid}
"""
        elif command_type == "D":
            # Delete link command
            return base_prompt + """
The 'edge' variable is an existing KGEdge object that will be deleted.
You should perform any necessary cleanup or validation before deletion.

Remember to populate the 'result' variable with a dictionary containing at least:
{'status': 'deleting', 'link_id': edge.uid}
"""

    return base_prompt


def _get_node_type_specific_guidance(node_type: str, command_type: str) -> str:
    """
    Get specific guidance for a node type and command.

    Provides additional instructions specific to each node type based on the operation.

    Args:
        node_type: The node type from NodeType enum
        command_type: The command type (C, U, D, or E)

    Returns:
        String with specific guidance for the node type
    """
    if node_type == NodeType.CHAT_MESSAGE:
        if command_type == "C":
            return """
For ChatMessage nodes, you should set:
- node.type = NodeType.CHAT_MESSAGE
- Set these properties in meta_props:
  - 'message_id': Generate or use provided message ID
  - 'chat_id': Get from context or instruction
  - 'role': 'assistant' or 'user'
  - 'content': The message content
  - 'timestamp': Current timestamp

If this is an assistant response, extract the appropriate response text 
from the instruction and set it as the content.
"""
        elif command_type == "E":
            return """
For ChatMessage evaluation, you should:
- Extract the message content and metadata
- Analyze sentiment, keywords, or other relevant aspects
- Return the analysis in the result dictionary
"""

    elif node_type == NodeType.SEARCH_CONFIG:
        if command_type == "C":
            return """
For SearchConfig nodes, you should set:
- node.type = NodeType.SEARCH_CONFIG
- Set these properties in meta_props:
  - 'config_id': Generate a new UUID
  - 'chat_id': Get from context
  - 'search_terms': List of search terms extracted from instruction
  - 'semantic_patterns': Optional list of patterns to look for
  - 'status': TaskStatus.PENDING
  - 'max_results': Default or specified number of results

Example structure:
node.meta_props = {
    'config_id': str(uuid.uuid4()),
    'chat_id': context.get('chat_id', 'unknown'),
    'search_terms': ['term1', 'term2'],
    'status': 'pending',
    'max_results': 15
}
"""
        elif command_type == "U":
            return """
For SearchConfig updates, you might:
- Update status (pending, running, completed, failed, cancelled)
- Add or modify search terms
- Update semantic patterns
- Set start_time or end_time
"""
        elif command_type == "E":
            return """
For SearchConfig evaluation, you should:
- Check the current status
- Validate search terms and parameters
- Return the current state and any validation results
"""

    elif node_type == NodeType.WEB_SEARCH:
        if command_type == "E":
            return """
For WebSearchFunction evaluation, extract:
- config_id: The ID of the SearchConfig to use
- Any additional parameters for the search

The WebSearchFunction requires a SearchConfig to operate. Your code should:
1. Extract the config_id from the instruction
2. Set up any additional parameters needed
3. Store these in the result dictionary:
   {'status': 'evaluating', 'function': 'web_search', 'config_id': config_id}

The actual search will be performed by the system, not in your code.
"""

    elif node_type == NodeType.PATTERN_EXTRACTION:
        if command_type == "E":
            return """
For PatternExtractionFunction evaluation, extract:
- config_id: The ID of the SearchConfig that produced results
- results: Optional list of result content to analyze directly

The PatternExtractionFunction needs either a config_id to find results,
or a direct list of results to analyze. Your code should extract these
parameters and store them in the result dictionary:
{'status': 'evaluating', 'function': 'pattern_extraction', 'config_id': config_id}
"""

    # Default guidance if no specific guidance is available
    return f"For {node_type} nodes, set appropriate properties based on the instruction."


def generate_natural_language_instruction(operation_type: str, entity_type: str,
                                          specific_type: Optional[str] = None,
                                          parameters: Dict[str, Any] = None) -> str:
    """
    Generate a natural language instruction for a specific operation.

    Creates clear, focused instructions for KGML operations based on the
    type of operation and entity, along with any specific parameters.

    Args:
        operation_type: Type of operation (Create, Update, Delete, Evaluate)
        entity_type: Type of entity (NODE or LINK)
        specific_type: Specific node or link type
        parameters: Dictionary of parameters for the operation

    Returns:
        Natural language instruction for the operation
    """
    parameters = parameters or {}

    # Map operation types to verbs
    operation_verbs = {
        "Create": "Create",
        "Update": "Update",
        "Delete": "Delete",
        "Evaluate": "Evaluate"
    }

    verb = operation_verbs.get(operation_type, operation_type)

    # Build the base instruction
    if entity_type == "NODE":
        base_instruction = f"{verb} a {specific_type or 'node'}"
    else:  # LINK
        base_instruction = f"{verb} a link"

    # Add parameter details
    if parameters:
        param_strings = []
        for key, value in parameters.items():
            if isinstance(value, list):
                param_strings.append(f"{key}=[{', '.join(repr(v) for v in value)}]")
            else:
                param_strings.append(f"{key}={repr(value)}")

        param_part = " with " + ", ".join(param_strings)
        base_instruction += param_part

    return base_instruction


def generate_search_results_prompt(config_id: str, results: List[str],
                                   patterns: List[str], chat_id: str) -> str:
    """
    Generate a prompt for summarizing search results.

    Creates a prompt that guides the model in generating a natural language
    summary of search results and patterns.

    Args:
        config_id: ID of the search configuration
        results: List of search result strings
        patterns: List of pattern strings
        chat_id: The chat ID

    Returns:
        A prompt for summarizing search results
    """
    # Keep prompt concise but informative
    prompt = f"""
Summarize these search results clearly and concisely.

CONFIG_ID: {config_id}
CHAT_ID: {chat_id}
RESULTS_COUNT: {len(results)}
PATTERNS_COUNT: {len(patterns)}

KEY PATTERNS:
"""

    # Add patterns (limit to 5 for focus)
    for i, pattern in enumerate(patterns[:5], 1):
        prompt += f"{i}. {pattern}\n"

    prompt += "\nSELECTED RESULTS:\n"

    # Add results (limit to 10 for focus)
    for i, result in enumerate(results[:10], 1):
        # Truncate long results
        max_length = 200
        result_text = result[:max_length] + "..." if len(result) > max_length else result
        prompt += f"{i}. {result_text}\n"

    prompt += """
RESPONSE GUIDELINES:
1. Start with a clear, direct summary of findings
2. Highlight 2-3 key insights from the patterns
3. Include 1-2 specific facts from the results
4. Keep response under 200 words
5. Use natural, conversational language
6. Format with markdown for readability
7. If results seem incomplete or need refinement, mention this

Format your response as if speaking directly to the user.
"""

    return prompt


def generate_intent_analysis_prompt(user_message: str, context: Dict[str, Any]) -> str:
    """
    Generate a prompt for analyzing user intent.

    Creates a prompt that guides the model in analyzing a user message
    to determine the intent and appropriate action.

    Args:
        user_message: The user's message
        context: Conversation context including history, active searches, etc.

    Returns:
        A prompt for intent analysis
    """
    # Extract relevant context elements
    recent_messages = context.get("recent_messages", [])
    active_configs = context.get("active_configs", [])
    completed_configs = context.get("completed_configs", [])
    entities = context.get("entities", {})

    # Create a concise but effective prompt
    prompt = f"""
Analyze this user message to determine intent and required action:

USER_MESSAGE: "{user_message}"

Intent types to consider:
- SEARCH: New search request
- FOLLOW_UP: Follow-up to previous search or conversation
- AUGMENT_SEARCH: Modify/extend existing search
- STATUS: Check status of search
- CANCEL: Cancel search
- HELP: Request for help
- CHAT: General conversation

"""

    # Add minimal but useful context
    if recent_messages:
        prompt += "RECENT_MESSAGES:\n"
        for i, msg in enumerate(recent_messages[-3:]):  # Last 3 messages only
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate long messages
            max_length = 100
            message_text = content[:max_length] + "..." if len(content) > max_length else content
            prompt += f"- {role.upper()}: {message_text}\n"

    if active_configs:
        prompt += "\nACTIVE_SEARCHES:\n"
        for config in active_configs[:2]:  # Only mention 2 most recent
            terms = ", ".join(config.get("search_terms", []))
            prompt += f"- {config.get('config_id')}: {terms}\n"

    if completed_configs:
        prompt += "\nCOMPLETED_SEARCHES:\n"
        for config in completed_configs[:2]:  # Only mention 2 most recent
            terms = ", ".join(config.get("search_terms", []))
            prompt += f"- {config.get('config_id')}: {terms}\n"

    if entities:
        prompt += "\nKEY_ENTITIES:\n"
        # Only include a few most relevant entities
        relevant_entities = list(entities.items())[:5]
        for entity, info in relevant_entities:
            prompt += f"- {entity} ({info.get('type', 'unknown')})\n"

    prompt += """
RESPONSE FORMAT:
INTENT: [intent_type]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [brief reason]
ACTION: [create_search|augment_search|answer_from_context|check_status|cancel_search|provide_help|chat_response]
PARAMETERS: 
search_terms: [list of terms]
config_id: [ID if relevant]
direct_response: [text for direct response if needed]
"""

    return prompt
