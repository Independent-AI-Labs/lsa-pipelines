"""
This module contains enhanced prompts for the LLM to use when analyzing user queries
and generating responses for the Private Web Search Assistant.
"""


def generate_intent_analysis_prompt(
        user_message: str,
        recent_messages: list,
        active_tasks: list = None,
        chat_tasks: list = None
) -> str:
    """
    Generate a prompt for the LLM to analyze the user message and determine
    the appropriate action to take.

    Args:
        user_message: The user's message
        recent_messages: Previous messages in the conversation
        active_tasks: List of currently active tasks (if any)
        chat_tasks: List of tasks associated with this chat

    Returns:
        A prompt for the LLM to analyze and determine actions
    """
    prompt = f"""
# Private Web Search Assistant: Intent Analysis

As the Private Web Search Assistant, analyze the user's message and determine the most appropriate action to take.

## User Message
"{user_message}"

## Available Actions
- help: User is asking for help or instructions
- answer_from_context: Answer using existing conversation without a new search
- create_search_task: Start a new search (provide search_terms, patterns, instructions)
- query_existing_task: User is asking about a specific existing task (provide task_id)
- cancel_task: Cancel a specific search task (provide task_id)
- cancel_all_tasks: Cancel all active search tasks
- show_results: Show results for a specific task or all completed tasks (optional task_id)
- show_status: Show the current status of tasks (optional task_id)

## Recent Conversation Context
"""

    # Add recent messages context
    if recent_messages and len(recent_messages) > 0:
        prompt += "Recent messages:\n"
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            prompt += f"- {role.upper()}: {content[:100]}{'...' if len(content) > 100 else ''}\n"
    else:
        prompt += "No recent messages available.\n"

    # Add information about active tasks if available
    if active_tasks and len(active_tasks) > 0:
        prompt += "\n## Active Tasks\n"
        for task in active_tasks:
            task_id = task.get('task_id', 'unknown')
            search_terms = task.get('search_terms', [])
            status = task.get('status', 'unknown')
            prompt += f"- Task {task_id}: {', '.join(search_terms)} (Status: {status})\n"
    else:
        prompt += "\n## Active Tasks\nNo active tasks.\n"

    # Add information about tasks associated with this chat
    if chat_tasks and len(chat_tasks) > 0:
        prompt += "\n## Tasks Associated With This Chat\n"
        for task_id in chat_tasks:
            prompt += f"- Task ID: {task_id}\n"
    else:
        prompt += "\n## Tasks Associated With This Chat\nNo tasks associated with this chat.\n"

    prompt += """
## Response Format
Please provide your analysis in a simple Markdown format:

```
ACTION: [action_name]
EXPLANATION: [brief explanation of why this action was chosen]
```

For create_search_task actions, add:
```
SEARCH_TERMS: [comma-separated list of search terms]
PATTERNS: [comma-separated list of patterns to look for, or "None"]
INSTRUCTIONS: [specific instructions for the search, or "None"]
```

For actions requiring a task_id, add:
```
TASK_ID: [specific task ID, or "None" if not specified]
```
"""

    return prompt


def generate_search_response_prompt(
        search_terms: list,
        patterns: list = None,
        instructions: str = None,
        task_id: str = None
) -> str:
    """
    Generate a prompt for the LLM to create a user-friendly response
    for a new search task.

    Args:
        search_terms: The search terms
        patterns: Optional list of patterns to look for
        instructions: Optional instructions for the search
        task_id: The task ID

    Returns:
        A prompt for the LLM to generate a response
    """
    prompt = f"""
# Private Web Search Assistant: Search Response

Create a friendly, informative response to the user about their new search request.

## Search Details
- Search Terms: {', '.join(search_terms)}
- Task ID: {task_id}
"""

    if patterns and len(patterns) > 0:
        prompt += f"- Patterns: {', '.join(patterns)}\n"
    else:
        prompt += "- Patterns: None specified\n"

    if instructions:
        prompt += f"- Instructions: {instructions}\n"
    else:
        prompt += "- Instructions: None specified\n"

    prompt += """
## Response Guidelines
- Be friendly and conversational
- Acknowledge the search request
- Mention that a dedicated chat has been created for this search
- Explain that all updates will be sent to the dedicated chat
- Mention that they can ask about this task from any chat using the task ID
- Keep the response concise (3-5 sentences)

## Response Format
Write a complete response that I can send directly to the user.
"""

    return prompt


def generate_context_answer_prompt(
        user_query: str,
        recent_messages: list,
        chat_tasks: list = None
) -> str:
    """
    Generate a prompt for the LLM to answer a query using existing context.

    Args:
        user_query: The user's query
        recent_messages: Recent messages in the conversation
        chat_tasks: Tasks associated with this chat

    Returns:
        A prompt for the LLM to generate an answer
    """
    prompt = f"""
# Private Web Search Assistant: Context-Based Answer

Determine if you can answer the user's query using the existing conversation context without needing a new web search.

## User Query
"{user_query}"

## Recent Conversation Context
"""

    # Add recent messages context
    if recent_messages and len(recent_messages) > 0:
        for msg in recent_messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n\n"
    else:
        prompt += "No recent conversation context available.\n"

    # Add information about tasks associated with this chat
    if chat_tasks and len(chat_tasks) > 0:
        prompt += "\n## Tasks Associated With This Chat\n"
        for task_id in chat_tasks:
            prompt += f"- Task ID: {task_id}\n"
    else:
        prompt += "\n## Tasks Associated With This Chat\nNo tasks associated with this chat.\n"

    prompt += """
## Response Format
First, determine if you can confidently answer the query based on the available context:

```
CAN_ANSWER: [Yes/No]
EXPLANATION: [brief explanation of why]
```

If CAN_ANSWER is Yes, add:
```
ANSWER: [your complete answer to the user]
```

If CAN_ANSWER is No, simply indicate that a web search might be needed.
"""

    return prompt


def generate_task_status_prompt(
        task: dict,
        is_dedicated_chat: bool = False
) -> str:
    """
    Generate a prompt for the LLM to create a status message for a task.

    Args:
        task: The task information
        is_dedicated_chat: Whether this is for the dedicated task chat

    Returns:
        A prompt for the LLM to generate a status message
    """
    prompt = f"""
# Private Web Search Assistant: Task Status

Create a user-friendly status message for the following task:

## Task Details
- Task ID: {task.get('task_id', 'unknown')}
- Search Terms: {', '.join(task.get('search_terms', []))}
- Status: {task.get('status', 'unknown')}
- Duration: {task.get('duration', 0):.1f} seconds
- Results Count: {len(task.get('results', []))}
- Patterns Count: {len(task.get('discovered_patterns', []))}
- Is Dedicated Chat: {'Yes' if is_dedicated_chat else 'No'}

## Response Guidelines
- Be friendly and conversational
- Provide clear information about the current status
- If completed, mention the number of results and patterns found
- If still running, mention the duration and that updates will continue
- If cancelled or failed, explain the status clearly
- Keep the response concise (2-4 sentences)

## Response Format
Write a complete response that I can send directly to the user.
"""

    return prompt


def generate_help_prompt() -> str:
    """
    Generate a prompt for the LLM to create a help message explaining
    the capabilities of the web search assistant.

    Returns:
        A prompt for the LLM to generate a help response
    """
    prompt = """
# Private Web Search Assistant: Help Message

Create a comprehensive yet concise help message explaining how to use the Private Web Search Assistant.

## Assistant Capabilities
- Performing web searches and presenting results
- Creating dedicated chats for each search task
- Tracking and managing multiple search tasks
- Analyzing search results for patterns
- Answering queries from existing context when possible
- Cancelling tasks and showing search progress

## Commands
- Basic search: The user can simply type their search query
- Advanced search: The user can provide search terms, patterns, and instructions
- Status checks: "status", "progress", or "status for task [ID]"
- View results: "show results" or "show results for [task ID]"
- Cancel tasks: "cancel task [task ID]" or "cancel all"
- Help: "help" or "?"

## Features to Highlight
- Each search task gets its own dedicated chat
- Users can ask about any task from any chat
- Advanced pattern matching and semantic analysis

## Response Guidelines
- Use clear formatting with headers and bullet points
- Include examples of advanced search syntax
- Make it scannable and easy to understand
- Use a friendly, helpful tone

## Response Format
Write a complete help message in Markdown format that I can send directly to the user.
"""

    return prompt


def generate_results_summary_prompt(
        task: dict,
        include_results: bool = True
) -> str:
    """
    Generate a prompt for the LLM to summarize the results of a search task.

    Args:
        task: The search task to summarize
        include_results: Whether to include full results in the prompt

    Returns:
        A prompt for the LLM to generate a summary
    """
    prompt = f"""
# Private Web Search Assistant: Results Summary

Create a concise, informative summary of the following search task and its results:

## Task Details
- Task ID: {task.get('task_id', 'unknown')}
- Search Query: {', '.join(task.get('search_terms', []))}
- Status: {task.get('status', 'unknown')}
- Duration: {task.get('duration', 0):.1f} seconds
"""

    # Include patterns if available
    patterns = task.get('discovered_patterns', [])
    if patterns:
        prompt += "\n## Discovered Patterns\n"
        for i, pattern in enumerate(patterns[:10], 1):  # Limit to 10 patterns
            prompt += f"{i}. {pattern}\n"
        if len(patterns) > 10:
            prompt += f"... and {len(patterns) - 10} more patterns\n"
    else:
        prompt += "\n## Discovered Patterns\nNo patterns discovered.\n"

    # Include results if requested and available
    if include_results:
        results = task.get('results', [])
        prompt += f"\n## Results ({len(results)})\n"

        if results:
            for i, result in enumerate(results[:15], 1):  # Limit to 15 results
                prompt += f"{i}. {result}\n"
            if len(results) > 15:
                prompt += f"... and {len(results) - 15} more results\n"
        else:
            prompt += "No results found.\n"

    prompt += """
## Response Guidelines
- Start with a brief recap of what was searched for
- Highlight the most important patterns discovered (if any)
- Summarize the most relevant results in a concise way
- Mention any limitations or suggestions for refining the search
- Format the response clearly with appropriate Markdown
- Be conversational and helpful

## Response Format
Write a complete summary that I can send directly to the user.
"""

    return prompt
