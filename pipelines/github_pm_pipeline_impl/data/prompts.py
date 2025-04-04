"""
Updated prompt templates for the GitHub Project Manager.
"""
from integration.github.models.project_models import ActionType
from integration.pipelines.pipelines.github_pm_pipeline_impl.data.pm_constants import (
    GITHUB_USER, GITHUB_ORG, TEST_REPOSITORY
)

# System prompt for the LLM
SYSTEM_PROMPT = f"""
You are an intelligent assistant specialized in managing GitHub Projects. 
Your primary responsibility is to understand user requests related to GitHub Projects and to execute the appropriate actions via the GitHub API.

Always follow these guidelines:
1. Understand the user's intent related to GitHub Projects
2. Decide on the most appropriate action based on the conversation context
3. Execute actions when needed and analyze the results
4. Provide clear, concise responses about what was done and what was found
5. Use Markdown for formatting responses to improve readability
6. Always include relevant details from API responses in your replies

Format your action decisions with special delimiters: ►ACTION_TYPE◄

Example formats:

For deciding to reply to the user:
►{ActionType.REPLY_TO_USER.value}◄

For deciding to get project information:
►{ActionType.GET_PROJECT_INFO.value}◄
```json
{{
  "owner": "{GITHUB_USER}",
  "project_number": 1
}}
```

For deciding to add an item to a project:
►{ActionType.ADD_ITEM_TO_PROJECT.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "content_id": "%CONTENT_ID%"
}}
```

For getting project information to update Category field:
►{ActionType.GET_PROJECT_FIELDS.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "field_name": "Category"
}}
```

For updating a select field:
►{ActionType.UPDATE_SELECT_FIELD.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "item_id": "%ITEM_ID%",
  "field_id": "%FIELD_ID%",
  "option_id": "%OPTION_ID%"
}}
```

Remember that all data must be in valid JSON format. Use placeholders with % symbols when you're referring to IDs that need to be resolved.
"""

# Context analysis prompt
CONTEXT_ANALYSIS_PROMPT = """
Analyze the following conversation history and current message to identify GitHub entities, 
projects, repositories, and other context. Extract specific names, numbers, and IDs.

Conversation history:
{conversation_history}

Current message:
{current_message}

Extract the following information if present:
- Entity: [user or organization name]
- Entity type: [User or Organization]
- Project ID: [if mentioned]
- Project number: [if mentioned]
- Project title: [if mentioned]
- Repository: [repo name]
- Repository owner: [owner name]
- Field: [field name such as Status, Category, Type]
- Field type: [text, select, etc.]
- Selected option: [option value]
- Issue types: [comma-separated list]
- Statuses: [comma-separated list]
- Categories: [comma-separated list]
- Labels: [comma-separated list]

Format your response in a simple key-value format, one per line. Only include information you can confidently extract.
"""

# Action descriptions for prompting
ACTION_DESCRIPTIONS = {
    ActionType.REPLY_TO_USER: "Reply directly to the user with information",
    ActionType.GET_PROJECT_INFO: "Get information about a specific GitHub Project",
    ActionType.GET_USER_PROJECTS: "Get all projects for a user",
    ActionType.GET_ORG_PROJECTS: "Get all projects for an organization",
    ActionType.GET_PROJECT_FIELDS: "Get fields defined in a project",
    ActionType.GET_PROJECT_ITEMS: "Get items in a project",
    ActionType.ADD_ITEM_TO_PROJECT: "Add an existing issue or PR to a project",
    ActionType.ADD_DRAFT_ISSUE: "Add a draft issue to a project",
    ActionType.UPDATE_PROJECT_SETTINGS: "Update project settings (title, description, etc.)",
    ActionType.UPDATE_TEXT_FIELD: "Update a text field value for a project item",
    ActionType.UPDATE_SELECT_FIELD: "Update a single select field value for a project item",
    ActionType.UPDATE_ITERATION_FIELD: "Update an iteration field value for a project item",
    ActionType.DELETE_PROJECT_ITEM: "Delete an item from a project",
    ActionType.CREATE_PROJECT: "Create a new GitHub Project",
    ActionType.CONVERT_DRAFT_TO_ISSUE: "Convert a draft issue to a real repository issue",
    ActionType.ADD_COMMENT_TO_ISSUE: "Add a comment to an existing issue",
    ActionType.CREATE_ISSUE: "Create a new issue in a repository",
    ActionType.GET_REPOSITORY_ID: "Get a repository's unique identifier"
}

# Action examples for prompting
ACTION_EXAMPLES = {
    ActionType.REPLY_TO_USER: f"""
►{ActionType.REPLY_TO_USER.value}◄
```json
{{
  "response": "Here's the information you requested about your projects."
}}
```
""",
    ActionType.GET_PROJECT_INFO: f"""
►{ActionType.GET_PROJECT_INFO.value}◄
```json
{{
  "owner": "{GITHUB_ORG}",
  "project_number": 5
}}
```
""",
    ActionType.GET_USER_PROJECTS: f"""
►{ActionType.GET_USER_PROJECTS.value}◄
```json
{{
  "username": "{GITHUB_USER}"
}}
```
""",
    ActionType.GET_ORG_PROJECTS: f"""
►{ActionType.GET_ORG_PROJECTS.value}◄
```json
{{
  "org_name": "{GITHUB_ORG}"
}}
```
""",
    ActionType.GET_PROJECT_FIELDS: f"""
►{ActionType.GET_PROJECT_FIELDS.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "field_name": "Category"
}}
```
""",
    ActionType.GET_PROJECT_ITEMS: f"""
►{ActionType.GET_PROJECT_ITEMS.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "item_title": "Business Plan"
}}
```
""",
    ActionType.ADD_ITEM_TO_PROJECT: f"""
►{ActionType.ADD_ITEM_TO_PROJECT.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "content_id": "%CONTENT_ID%"
}}
```
""",
    ActionType.ADD_DRAFT_ISSUE: f"""
►{ActionType.ADD_DRAFT_ISSUE.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "title": "Fix navigation bug",
  "body": "The navigation menu is not working correctly on mobile.",
  "category": "SOLUTIONS"
}}
```
""",
    ActionType.UPDATE_PROJECT_SETTINGS: f"""
►{ActionType.UPDATE_PROJECT_SETTINGS.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "title": "New Project Title",
  "public": true,
  "short_description": "Updated project description"
}}
```
""",
    ActionType.UPDATE_TEXT_FIELD: f"""
►{ActionType.UPDATE_TEXT_FIELD.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "item_id": "%ITEM_ID%",
  "field_id": "%FIELD_ID%",
  "text_value": "Updated text value"
}}
```
""",
    ActionType.UPDATE_SELECT_FIELD: f"""
►{ActionType.UPDATE_SELECT_FIELD.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "item_id": "%ITEM_ID%",
  "field_id": "%FIELD_ID%",
  "option_id": "%OPTION_ID%"
}}
```
""",
    ActionType.UPDATE_ITERATION_FIELD: f"""
►{ActionType.UPDATE_ITERATION_FIELD.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "item_id": "%ITEM_ID%",
  "field_id": "%FIELD_ID%",
  "iteration_id": "%ITERATION_ID%"
}}
```
""",
    ActionType.DELETE_PROJECT_ITEM: f"""
►{ActionType.DELETE_PROJECT_ITEM.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "item_id": "%ITEM_ID%"
}}
```
""",
    ActionType.CREATE_PROJECT: f"""
►{ActionType.CREATE_PROJECT.value}◄
```json
{{
  "owner_id": "%OWNER_ID%",
  "title": "New Development Roadmap"
}}
```
""",
    ActionType.CONVERT_DRAFT_TO_ISSUE: f"""
►{ActionType.CONVERT_DRAFT_TO_ISSUE.value}◄
```json
{{
  "project_id": "%PROJECT_ID%",
  "draft_item_id": "%ITEM_ID%",
  "owner": "{GITHUB_USER}",
  "repo": "{TEST_REPOSITORY}",
  "labels": ["bug", "enhancement"]
}}
```
""",
    ActionType.ADD_COMMENT_TO_ISSUE: f"""
►{ActionType.ADD_COMMENT_TO_ISSUE.value}◄
```json
{{
  "owner": "{GITHUB_USER}",
  "repo": "{TEST_REPOSITORY}",
  "issue_number": 42,
  "body": "This is a comment added via the API"
}}
```
""",
    ActionType.CREATE_ISSUE: f"""
►{ActionType.CREATE_ISSUE.value}◄
```json
{{
  "owner": "{GITHUB_USER}",
  "repo": "{TEST_REPOSITORY}",
  "title": "New issue title",
  "body": "Issue description",
  "labels": ["bug", "feature"]
}}
```
""",
    ActionType.GET_REPOSITORY_ID: f"""
►{ActionType.GET_REPOSITORY_ID.value}◄
```json
{{
  "owner": "{GITHUB_USER}",
  "repo": "{TEST_REPOSITORY}"
}}
```
"""
}


# Generate action list for prompts
def generate_action_list():
    """Generate a formatted list of actions with descriptions and examples."""
    action_list = ""
    for i, action_type in enumerate(ActionType, start=1):
        action_list += f"{i}. {ACTION_DESCRIPTIONS[action_type]} ({action_type.value})\n"
        action_list += f"Example:\n{ACTION_EXAMPLES[action_type]}\n"
    return action_list


# Prompt template for deciding on an action
ACTION_DECISION_PROMPT = """
Based on the following conversation history, current message, and context information, decide which action to take next.

Conversation history:
{conversation_history}

Current message:
{current_message}

{context_info}

Analyze the user's intent carefully. Consider any project-specific custom fields, issue types, statuses, and categories from the context.

If the context indicates custom project fields:
- Use those field names and options when planning your action
- Refer to the available options for select fields
- Remember that each project may have different configurations

If the user is asking about GitHub Projects, you should decide which action to take next.

Determine the most appropriate action and output ONLY the action type with its parameters using the format shown above.
Always use JSON format for parameters inside code blocks.
Use proper placeholders (e.g., %PROJECT_ID%, %FIELD_ID%, %ITEM_ID%) for values that need to be resolved later.
"""

# Prompt template for analyzing action results
RESULT_ANALYSIS_PROMPT = """
Analyze the results of the action that was just executed to determine the next action.

Action executed:
Type: {action_type}
Parameters: {action_parameters}

Result:
Success: {result_success}
Data: {result_data}
Message: {result_message}

Context Information:
{context_info}

Previous actions:
{action_history}

Conversation History:
{conversation_history}

Consider all project-specific custom fields, issue types, statuses, and categories from the context when deciding the next action.

Based on this information, decide what to do next. You have two options:

1. Reply to the user if you have all the information needed:
►REPLY_TO_USER◄
```json
{
  "response": "Your detailed response here"
}
```

2. Take another action if you need more information or need to perform additional operations.

If you believe we should stop the action chain and provide a direct response to the user, include the text "STOP_ACTION_CHAIN" in your response.

Determine the most appropriate next action and output ONLY the action type with its parameters using the format shown above.
"""

# Prompt template for generating user reply
USER_REPLY_PROMPT = """
Generate a helpful response to the user based on the conversation history and the actions that were executed.

Conversation history:
{conversation_history}

Actions executed:
{action_history}

Context Information:
{context_info}

Your response should:
1. Be clear and concise
2. Use Markdown for formatting where appropriate
3. Explain what was done (if actions were executed)
4. Include relevant information from the action results
5. Answer the user's original question or fulfill their request
6. Be conversational and helpful
7. Include any field IDs, project IDs, or other technical information that might be helpful for future requests
8. Reference project-specific custom fields, issue types, statuses, and categories when relevant

Generate a complete response that the user will see.
"""

# Example workflow prompts for specific action chains
GET_PROJECT_WORKFLOW = """
To get information about a GitHub Project, I need:
1. The owner (username or organization name)
2. The project number (visible in the URL)

With this information, I'll:
1. First verify if the owner exists
2. Get the project details
3. If successful, get the project fields and items

Let me execute these steps for you.
"""

ADD_ITEM_WORKFLOW = """
To add an item to a GitHub Project, I need:
1. The project ID (which I can help find)
2. The content ID (for issues or PRs)

I'll follow these steps:
1. Find the project ID using the owner and project number
2. Add the item to the project
3. Confirm the item was added successfully

Let me execute these steps for you.
"""

CONVERT_DRAFT_WORKFLOW = """
To convert a draft issue to a real issue, I need:
1. The project ID
2. The draft item ID
3. A repository to create the issue in

I'll follow these steps:
1. Get the draft issue details
2. Create a new issue in the repository
3. Remove the draft from the project
4. Add the new issue to the project

Let me execute these steps for you.
"""

# Dictionary of required parameters for each action type
REQUIRED_PARAMETERS = {
    ActionType.REPLY_TO_USER: [],
    ActionType.GET_PROJECT_INFO: ["owner", "project_number"],
    ActionType.GET_USER_PROJECTS: ["username"],
    ActionType.GET_ORG_PROJECTS: ["org_name"],
    ActionType.GET_PROJECT_FIELDS: ["project_id"],
    ActionType.GET_PROJECT_ITEMS: ["project_id"],
    ActionType.ADD_ITEM_TO_PROJECT: ["project_id", "content_id"],
    ActionType.ADD_DRAFT_ISSUE: ["project_id", "title"],
    ActionType.UPDATE_PROJECT_SETTINGS: ["project_id"],
    ActionType.UPDATE_TEXT_FIELD: ["project_id", "item_id", "field_id", "text_value"],
    ActionType.UPDATE_SELECT_FIELD: ["project_id", "item_id", "field_id", "option_id"],
    ActionType.UPDATE_ITERATION_FIELD: ["project_id", "item_id", "field_id", "iteration_id"],
    ActionType.DELETE_PROJECT_ITEM: ["project_id", "item_id"],
    ActionType.CREATE_PROJECT: ["owner_id", "title"],
    ActionType.CONVERT_DRAFT_TO_ISSUE: ["project_id", "draft_item_id", "owner", "repo"],
    ActionType.ADD_COMMENT_TO_ISSUE: ["owner", "repo", "issue_number", "body"],
    ActionType.CREATE_ISSUE: ["owner", "repo", "title", "body"],
    ActionType.GET_REPOSITORY_ID: ["owner", "repo"]
}

# Dictionary of parameter descriptions for user-friendly messages
PARAMETER_DESCRIPTIONS = {
    "owner": "the GitHub username or organization name",
    "project_number": "the project number (visible in the GitHub URL)",
    "username": "the GitHub username",
    "org_name": "the GitHub organization name",
    "project_id": "the project's unique identifier",
    "content_id": "the content ID (for issues or PRs)",
    "item_id": "the project item's ID",
    "field_id": "the field's ID",
    "text_value": "the text value to set",
    "option_id": "the ID of the option to select",
    "iteration_id": "the ID of the iteration",
    "title": "the title text",
    "body": "the body text",
    "owner_id": "the owner's unique identifier",
    "public": "whether the project is public (true/false)",
    "readme": "the project's README content",
    "short_description": "a short description of the project",
    "limit": "the maximum number of items to return",
    "filter_query": "text to filter projects by name",
    "draft_item_id": "the ID of the draft item to convert",
    "repo": "the repository name",
    "issue_number": "the issue number",
    "labels": "comma-separated list of labels to apply",
    "field_name": "the name of the field",
    "item_title": "the title of the item",
    "selected_option_name": "the name of the option to select",
    "category": "the category to assign"
}

# Error messages for user-friendly responses
ERROR_MESSAGES = {
    "BASE": "I encountered an issue while trying to ",
    ActionType.GET_PROJECT_INFO: "retrieve the project information.",
    ActionType.GET_USER_PROJECTS: "retrieve the user's projects.",
    ActionType.GET_ORG_PROJECTS: "retrieve the organization's projects.",
    ActionType.GET_PROJECT_FIELDS: "retrieve the project fields.",
    ActionType.GET_PROJECT_ITEMS: "retrieve the project items.",
    ActionType.ADD_ITEM_TO_PROJECT: "add the item to the project.",
    ActionType.ADD_DRAFT_ISSUE: "add the draft issue to the project.",
    ActionType.UPDATE_PROJECT_SETTINGS: "update the project settings.",
    ActionType.UPDATE_TEXT_FIELD: "update the text field.",
    ActionType.UPDATE_SELECT_FIELD: "update the select field.",
    ActionType.UPDATE_ITERATION_FIELD: "update the iteration field.",
    ActionType.DELETE_PROJECT_ITEM: "delete the item from the project.",
    ActionType.CREATE_PROJECT: "create the new project.",
    ActionType.CONVERT_DRAFT_TO_ISSUE: "convert the draft issue to a real issue.",
    ActionType.ADD_COMMENT_TO_ISSUE: "add a comment to the issue.",
    ActionType.CREATE_ISSUE: "create a new issue.",
    ActionType.GET_REPOSITORY_ID: "retrieve the repository information.",
    "RATE_LIMIT": "GitHub API rate limit exceeded. Please try again later.",
    "NO_ENTITY": "No entity name provided and no defaults configured.",
    "ENTITY_NOT_FOUND": "Could not find GitHub user or organization: {entity}",

    # Suggestions for specific action types
    f"{ActionType.GET_PROJECT_INFO.value}_SUGGESTIONS": [
        "Make sure the project exists and is accessible to you.",
        "Check if the project owner and number are correct.",
        "Ensure the GitHub token has sufficient permissions."
    ],
    f"{ActionType.GET_USER_PROJECTS.value}_SUGGESTIONS": [
        "Make sure the username is correct.",
        "Check if the user has public projects.",
        "Ensure the GitHub token has sufficient permissions."
    ],
    f"{ActionType.GET_ORG_PROJECTS.value}_SUGGESTIONS": [
        "Make sure the organization name is correct.",
        "Check if the organization has public projects.",
        "Ensure the GitHub token has sufficient permissions."
    ],
    f"{ActionType.ADD_ITEM_TO_PROJECT.value}_SUGGESTIONS": [
        "Make sure the project ID and content ID are correct.",
        "Check if you have write access to the project.",
        "Ensure the item hasn't already been added to the project."
    ],
    f"{ActionType.ADD_DRAFT_ISSUE.value}_SUGGESTIONS": [
        "Make sure the project ID is correct.",
        "Check if you have write access to the project.",
        "Ensure the title and body are properly formatted."
    ],
    f"{ActionType.CONVERT_DRAFT_TO_ISSUE.value}_SUGGESTIONS": [
        "Make sure the project ID and draft item ID are correct.",
        "Check if the repository exists and you have write access to it.",
        "Ensure the draft issue contains valid content."
    ],
    f"{ActionType.ADD_COMMENT_TO_ISSUE.value}_SUGGESTIONS": [
        "Make sure the repository and issue number are correct.",
        "Check if you have write access to the repository.",
        "Ensure the comment body is properly formatted."
    ],
    f"{ActionType.CREATE_ISSUE.value}_SUGGESTIONS": [
        "Make sure the repository name and owner are correct.",
        "Check if you have write access to the repository.",
        "Ensure the issue title and body are properly formatted."
    ],
    f"{ActionType.UPDATE_SELECT_FIELD.value}_SUGGESTIONS": [
        "Make sure all the IDs (project, item, field, option) are correct.",
        "Check if the field is actually a select field.",
        "Ensure the option ID exists in this field's options."
    ],
    "DEFAULT_SUGGESTIONS": [
        "Make sure all parameters are correct.",
        "Check if you have the necessary permissions.",
        "Try again with more specific information."
    ]
}

# Success messages for user-friendly responses
SUCCESS_MESSAGES = {
    "BASE": "I successfully ",
    ActionType.GET_PROJECT_INFO: "retrieved information about the project '{title}' owned by {owner}.",
    ActionType.GET_USER_PROJECTS: "found {count} projects for the user {username}.",
    ActionType.GET_ORG_PROJECTS: "found {count} projects for the organization {org_name}.",
    ActionType.GET_PROJECT_FIELDS: "retrieved all fields for the project.",
    ActionType.GET_PROJECT_ITEMS: "retrieved items from the project.",
    ActionType.ADD_ITEM_TO_PROJECT: "added the item to the project.",
    ActionType.ADD_DRAFT_ISSUE: "added a draft issue to the project.",
    ActionType.UPDATE_PROJECT_SETTINGS: "updated the project settings.",
    ActionType.UPDATE_TEXT_FIELD: "updated the text field value.",
    ActionType.UPDATE_SELECT_FIELD: "updated the select field value.",
    ActionType.UPDATE_ITERATION_FIELD: "updated the iteration field value.",
    ActionType.DELETE_PROJECT_ITEM: "deleted the item from the project.",
    ActionType.CREATE_PROJECT: "created a new project.",
    ActionType.CONVERT_DRAFT_TO_ISSUE: "converted the draft issue to a real issue in the {repo} repository.",
    ActionType.ADD_COMMENT_TO_ISSUE: "added a comment to issue #{issue_number} in the {repo} repository.",
    ActionType.CREATE_ISSUE: "created a new issue #{issue_number} in the {repo} repository.",
    ActionType.GET_REPOSITORY_ID: "retrieved information about the {repo} repository."
}
