import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Generator, Iterator, List, Union, Tuple

from pydantic import BaseModel

from integration.github.github_client import GitHubClient
from integration.github.models.project_models import Action, ActionType, ActionResult
from integration.net.ollama.ollama_api import prompt_model
# Constants and configuration imports
from integration.pipelines.pipelines.github_pm_pipeline_impl.data.pm_constants import (
    GITHUB_TOKEN,
    DEFAULT_LLM,
    BASE_API_URL,
    API_RATE_LIMIT,
    MAX_ACTIONS,
    LOG_LEVEL,
    LOG_FORMAT,
    DEFAULT_LABELS,
    TEST_REPOSITORY
)
from integration.pipelines.pipelines.github_pm_pipeline_impl.data.prompts import (
    SYSTEM_PROMPT,
    ACTION_DECISION_PROMPT,
    RESULT_ANALYSIS_PROMPT,
    USER_REPLY_PROMPT,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    PARAMETER_DESCRIPTIONS,
    REQUIRED_PARAMETERS
)


class ChatInfo(BaseModel):
    """Model for chat information."""
    chat_id: str
    user_id: str


class Pipeline:
    """
    LLM-based GitHub Project Manager Pipeline.

    This pipeline analyzes user messages, decides on appropriate GitHub
    Project actions, executes them, and provides conversational responses.
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.name = "GitHub Project Manager"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(LOG_LEVEL)

        # Set up logging format
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Initialize GitHub client
        self.gh_client = GitHubClient(GITHUB_TOKEN, BASE_API_URL)

        # Model configuration
        self.default_model = DEFAULT_LLM

        # Track rate limit information
        self.rate_limit_info = {
            "limit": API_RATE_LIMIT,
            "remaining": API_RATE_LIMIT,
            "reset_time": None,
            "last_updated": time.time()
        }

    async def on_startup(self):
        """Initialize resources when the server starts."""
        self.logger.info(f"on_startup:{__name__}")
        # Initialize rate limit info
        self._refresh_rate_limit_info()

    async def on_shutdown(self):
        """Clean up resources when the server stops."""
        self.logger.info(f"on_shutdown:{__name__}")
        # Any cleanup code here

    def pipe(self, user_message: str, model_id: str, messages: List[Dict], body: Dict[str, Any]) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function that processes incoming messages and returns a response.

        Args:
            user_message: The user's message
            model_id: The model ID (pipeline ID)
            messages: The message history
            body: The request body

        Returns:
            The response
        """
        self.logger.info(f"pipe:{__name__}")

        # Check if this is just a title request
        if body.get("title", False):
            return self.name

        # Ignore autocomplete requests
        if user_message.startswith("### Task:"):
            return ""

        # Extract chat info
        chat_info = self._extract_chat_info(body)
        chat_id = chat_info.chat_id
        user_id = chat_info.user_id

        yield "<think>"
        yield "Analyzing user message..."

        # Store the pipeline model_id for dedicated chats and use default_model for LLM operations
        pipeline_model_id = model_id  # This is the model that represents the pipeline itself
        llm = self.default_model  # This is what we'll use for actual LLM operations

        try:
            # Process conversation history to decide on action
            action = self._decide_action(user_message, messages, llm)
            yield f"\nAction decided: {action.type.value}"

            # Execute action chain until we have a final response
            final_response = self._execute_action_chain(action, chat_id, user_id, messages, llm)

            # Return the response
            yield from self._finish_response(final_response)

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            error_response = f"I encountered an error while processing your request: {str(e)}"
            yield from self._finish_response(error_response)

    def _extract_chat_info(self, body: Dict[str, Any]) -> ChatInfo:
        """Extract chat information from the request body."""
        return ChatInfo(
            chat_id=body.get("chat_id", "unknown"),
            user_id=body.get("user_id", "unknown")
        )

    def _decide_action(self, user_message: str, messages: List[Dict], model: str) -> Action:
        """
        Analyze the user message and conversation history to decide on an action.

        Returns:
            Action: The decided action to take
        """
        # Format conversation history for the LLM
        formatted_history = self._format_conversation_history(messages)

        # Prepare prompt for action decision
        prompt = ACTION_DECISION_PROMPT.format(
            conversation_history=formatted_history,
            current_message=user_message
        )

        # Get LLM decision
        response = prompt_model(prompt, model=model, system_prompt=SYSTEM_PROMPT)

        # Parse the response to get the decided action
        # The response should be in the format: ►ACTION_TYPE◄ followed by parameters
        action_type, *params = self._parse_action_response(response)

        return Action(
            type=action_type,
            parameters=params[0] if params else {}
        )

    def _execute_action_chain(self, initial_action: Action, chat_id: str, user_id: str,
                              messages: List[Dict], model: str) -> str:
        """
        Execute a chain of actions until we have a final response to the user.

        Returns:
            str: The final response to send to the user
        """
        current_action = initial_action
        action_history = []

        # Execute actions until we reach REPLY_TO_USER action or hit the safety limit
        while current_action.type != ActionType.REPLY_TO_USER and len(action_history) < MAX_ACTIONS:
            self.logger.info(f"Executing action {len(action_history) + 1}: {current_action.type.value}")

            # Validate parameters before executing
            is_valid, missing_params, validation_message = self._validate_parameters(current_action)

            if not is_valid:
                # If parameters are invalid, add to history and move to REPLY_TO_USER with error
                action_result = ActionResult(
                    success=False,
                    data={"missing_params": missing_params},
                    message=validation_message
                )
                action_history.append((current_action, action_result))

                # Set next action to reply to user with the error
                current_action = Action(
                    type=ActionType.REPLY_TO_USER,
                    parameters={"error": validation_message}
                )
                continue

            # Execute the current action
            result = self._execute_action(current_action, chat_id, user_id)
            action_history.append((current_action, result))

            # If this is a project discovery action and we need to filter by name
            if (current_action.type in [ActionType.GET_USER_PROJECTS, ActionType.GET_ORG_PROJECTS] and
                    "filter_query" in current_action.parameters and
                    result.success):
                # Filter projects by name
                filter_query = current_action.parameters["filter_query"]
                projects = result.data.get("projects", [])
                filtered_projects = [
                    p for p in projects
                    if filter_query.lower() in p.get("title", "").lower()
                ]

                # Update the result with filtered projects
                result.data["projects"] = filtered_projects
                result.message = f"Found {len(filtered_projects)} projects matching '{filter_query}'"

                # Replace the last action history entry with the updated result
                action_history[-1] = (current_action, result)

            # Handle rate limiting
            if not result.success and "rate limit" in result.message.lower():
                # Create a user-friendly rate limit message
                rate_limit_message = ERROR_MESSAGES["RATE_LIMIT"]

                # Set next action to reply to user with the rate limit message
                current_action = Action(
                    type=ActionType.REPLY_TO_USER,
                    parameters={"error": rate_limit_message}
                )
                continue

            # Analyze the result to decide on the next action
            current_action = self._analyze_result(current_action, result, action_history, messages, model)

        # Safety check for action chain length
        if len(action_history) >= MAX_ACTIONS:
            summary = self._summarize_action_history(action_history)
            return (f"I've executed several actions but couldn't completely resolve your request. "
                    f"Here's what I found so far:\n\n{summary}")

        # Check if the final action has an error parameter (from parameter validation)
        if current_action.type == ActionType.REPLY_TO_USER and "error" in current_action.parameters:
            error_message = current_action.parameters["error"]
            # Generate a more user-friendly error message
            last_action, last_result = action_history[-1] if action_history else (None, None)
            if last_action:
                return self._generate_user_friendly_error(last_action.type, error_message)
            else:
                return f"I encountered an issue: {error_message}"

        # Generate user reply based on action history
        return self._generate_user_reply(action_history, messages, model)

    def _execute_action(self, action: Action, chat_id: str, user_id: str) -> ActionResult:
        """
        Execute a GitHub API action and return the result.

        Parameters:
            action (Action): The action to execute
            chat_id (str): The current chat ID
            user_id (str): The current user ID

        Returns:
            ActionResult: The result of the action
        """
        self.logger.info(f"Executing action: {action.type.value} with parameters: {action.parameters}")

        try:
            # Handle different action types
            if action.type == ActionType.GET_PROJECT_INFO:
                return self.gh_client.get_project_info(
                    owner=action.parameters.get("owner"),
                    project_number=int(action.parameters.get("project_number"))
                )

            elif action.type == ActionType.GET_USER_PROJECTS:
                limit = int(action.parameters.get("limit", 20))
                return self.gh_client.get_user_projects(
                    username=action.parameters.get("username"),
                    limit=limit
                )

            elif action.type == ActionType.GET_ORG_PROJECTS:
                limit = int(action.parameters.get("limit", 20))
                return self.gh_client.get_org_projects(
                    org_name=action.parameters.get("org_name"),
                    limit=limit
                )

            elif action.type == ActionType.GET_PROJECT_FIELDS:
                return self.gh_client.get_project_fields(
                    project_id=action.parameters.get("project_id")
                )

            elif action.type == ActionType.GET_PROJECT_ITEMS:
                limit = int(action.parameters.get("limit", 20))
                return self.gh_client.get_project_items(
                    project_id=action.parameters.get("project_id"),
                    limit=limit
                )

            elif action.type == ActionType.ADD_ITEM_TO_PROJECT:
                return self.gh_client.add_item_to_project(
                    project_id=action.parameters.get("project_id"),
                    content_id=action.parameters.get("content_id")
                )

            elif action.type == ActionType.ADD_DRAFT_ISSUE:
                return self.gh_client.add_draft_issue(
                    project_id=action.parameters.get("project_id"),
                    title=action.parameters.get("title"),
                    body=action.parameters.get("body", "")
                )

            elif action.type == ActionType.UPDATE_PROJECT_SETTINGS:
                return self.gh_client.update_project_settings(
                    project_id=action.parameters.get("project_id"),
                    title=action.parameters.get("title"),
                    public=action.parameters.get("public"),
                    readme=action.parameters.get("readme"),
                    short_description=action.parameters.get("short_description")
                )

            elif action.type == ActionType.UPDATE_TEXT_FIELD:
                return self.gh_client.update_text_field(
                    project_id=action.parameters.get("project_id"),
                    item_id=action.parameters.get("item_id"),
                    field_id=action.parameters.get("field_id"),
                    text_value=action.parameters.get("text_value")
                )

            elif action.type == ActionType.UPDATE_SELECT_FIELD:
                return self.gh_client.update_select_field(
                    project_id=action.parameters.get("project_id"),
                    item_id=action.parameters.get("item_id"),
                    field_id=action.parameters.get("field_id"),
                    option_id=action.parameters.get("option_id")
                )

            elif action.type == ActionType.UPDATE_ITERATION_FIELD:
                return self.gh_client.update_iteration_field(
                    project_id=action.parameters.get("project_id"),
                    item_id=action.parameters.get("item_id"),
                    field_id=action.parameters.get("field_id"),
                    iteration_id=action.parameters.get("iteration_id")
                )

            elif action.type == ActionType.DELETE_PROJECT_ITEM:
                return self.gh_client.delete_project_item(
                    project_id=action.parameters.get("project_id"),
                    item_id=action.parameters.get("item_id")
                )

            elif action.type == ActionType.CREATE_PROJECT:
                return self.gh_client.create_project(
                    owner_id=action.parameters.get("owner_id"),
                    title=action.parameters.get("title")
                )

            elif action.type == ActionType.CONVERT_DRAFT_TO_ISSUE:
                # Parse labels if provided
                labels = None
                if "labels" in action.parameters:
                    if isinstance(action.parameters["labels"], str):
                        labels = [label.strip() for label in action.parameters["labels"].split(",") if label.strip()]
                    elif isinstance(action.parameters["labels"], list):
                        labels = action.parameters["labels"]

                # Use default labels if none provided and defaults exist
                if not labels and DEFAULT_LABELS:
                    labels = DEFAULT_LABELS

                return self.gh_client.convert_draft_to_issue(
                    project_id=action.parameters.get("project_id"),
                    draft_item_id=action.parameters.get("draft_item_id"),
                    owner=action.parameters.get("owner"),
                    repo=action.parameters.get("repo"),
                    labels=labels
                )

            elif action.type == ActionType.ADD_COMMENT_TO_ISSUE:
                return self.gh_client.add_comment_to_issue(
                    owner=action.parameters.get("owner"),
                    repo=action.parameters.get("repo"),
                    issue_number=int(action.parameters.get("issue_number")),
                    body=action.parameters.get("body")
                )

            elif action.type == ActionType.CREATE_ISSUE:
                # Parse labels if provided
                labels = None
                if "labels" in action.parameters:
                    if isinstance(action.parameters["labels"], str):
                        labels = [label.strip() for label in action.parameters["labels"].split(",") if label.strip()]
                    elif isinstance(action.parameters["labels"], list):
                        labels = action.parameters["labels"]

                # Use default labels if none provided and defaults exist
                if not labels and DEFAULT_LABELS:
                    labels = DEFAULT_LABELS

                return self.gh_client.create_issue(
                    owner=action.parameters.get("owner"),
                    repo=action.parameters.get("repo"),
                    title=action.parameters.get("title"),
                    body=action.parameters.get("body", ""),
                    labels=labels
                )

            elif action.type == ActionType.GET_REPOSITORY_ID:
                return self.gh_client.get_repository_id(
                    owner=action.parameters.get("owner"),
                    repo=action.parameters.get("repo")
                )

            elif action.type == ActionType.REPLY_TO_USER:
                # This action doesn't need GitHub API calls - it's handled in _generate_user_reply
                return ActionResult(
                    success=True,
                    data={},
                    message="Ready to generate user reply"
                )

            else:
                return ActionResult(
                    success=False,
                    data={},
                    message=f"Unknown action type: {action.type.value}"
                )

        except Exception as e:
            self.logger.error(f"Error executing action {action.type.value}: {str(e)}")
            return ActionResult(
                success=False,
                data={},
                message=f"Failed to execute {action.type.value}: {str(e)}"
            )

    def _analyze_result(self, action: Action, result: ActionResult,
                        action_history: List, messages: List[Dict], model: str) -> Action:
        """
        Analyze an action result to decide on the next action.

        Returns:
            Action: The next action to take
        """
        # Format action history for the LLM
        formatted_history = self._format_action_history(action_history)

        # Prepare prompt for result analysis
        prompt = RESULT_ANALYSIS_PROMPT.format(
            action_type=action.type.value,
            action_parameters=action.parameters,
            result_success=result.success,
            result_data=result.data,
            result_message=result.message,
            action_history=formatted_history
        )

        # Get LLM decision on next action
        response = prompt_model(prompt, model=model, system_prompt=SYSTEM_PROMPT)

        # Parse the response to get the next action
        action_type, *params = self._parse_action_response(response)

        return Action(
            type=action_type,
            parameters=params[0] if params else {}
        )

    def _generate_user_reply(self, action_history: List, messages: List[Dict], model: str) -> str:
        """
        Generate a reply to the user based on the action history.

        Returns:
            str: The reply to send to the user
        """
        # Format conversation and action history for the LLM
        formatted_conv_history = self._format_conversation_history(messages)
        formatted_action_history = self._format_action_history(action_history)

        # Prepare prompt for user reply
        prompt = USER_REPLY_PROMPT.format(
            conversation_history=formatted_conv_history,
            action_history=formatted_action_history
        )

        # Get LLM to generate the reply
        response = prompt_model(prompt, model=model, system_prompt=SYSTEM_PROMPT)

        return response

    def _format_conversation_history(self, messages: List[Dict]) -> str:
        """Format the conversation history for LLM prompts."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted += f"{role.upper()}: {content}\n\n"
        return formatted

    def _format_action_history(self, action_history: List) -> str:
        """Format the action history for LLM prompts."""
        formatted = ""
        for i, (action, result) in enumerate(action_history):
            formatted += f"ACTION {i + 1}: {action.type.value}\n"
            formatted += f"PARAMETERS: {action.parameters}\n"
            formatted += f"RESULT: {'Success' if result.success else 'Failed'}\n"
            formatted += f"DATA: {result.data}\n"
            formatted += f"MESSAGE: {result.message}\n\n"
        return formatted

    def _parse_action_response(self, response: str) -> tuple:
        """
        Parse the LLM response to extract action type and parameters.

        The response should be in the format: ►ACTION_TYPE◄ followed by parameters

        Returns:
            tuple: (ActionType, parameters dict)
        """
        # Extract action type between ► and ◄
        action_match = re.search(r'►([A-Z_]+)◄', response)
        if not action_match:
            # Default to REPLY_TO_USER if no action is found
            return ActionType.REPLY_TO_USER, {}

        action_type_str = action_match.group(1)
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            # If the action type is invalid, default to REPLY_TO_USER
            self.logger.warning(f"Invalid action type: {action_type_str}, defaulting to REPLY_TO_USER")
            return ActionType.REPLY_TO_USER, {}

        # Extract parameters if any (should be in markdown code blocks)
        parameters = {}
        params_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if params_match:
            try:
                parameters = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract key-value pairs
                param_text = params_match.group(1)
                for line in param_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        parameters[key.strip()] = value.strip()
        else:
            # Try to extract key-value pairs from the plain text response
            lines = response.split('\n')
            for line in lines:
                if ':' in line and not line.startswith(('ACTION', 'RESULT', 'DATA', 'MESSAGE')):
                    key, value = line.split(':', 1)
                    parameters[key.strip()] = value.strip()

        return action_type, parameters

    def _summarize_action_history(self, action_history: List) -> str:
        """Create a user-friendly summary of the action history."""
        summary = ""
        for i, (action, result) in enumerate(action_history):
            summary += f"- {action.type.value}: "
            if result.success:
                summary += f"Succeeded. {result.message}\n"
            else:
                summary += f"Failed. {result.message}\n"
        return summary

    def _finish_response(self, response: str) -> Iterator[str]:
        """Finish the response by closing the thinking tag and yielding the response."""
        yield "</think>"
        yield response

    def _handle_api_rate_limit(self, response: Dict[str, Any]) -> Tuple[bool, Dict]:
        """
        Check if the response indicates a rate limit error and handle appropriately.

        Parameters:
            response (Dict): The response from the GitHub API

        Returns:
            Tuple[bool, Dict]: (is_rate_limited, rate_limit_info)
        """
        # Check for rate limit errors in GraphQL response
        if "errors" in response:
            for error in response.get("errors", []):
                if "type" in error and error["type"] == "RATE_LIMITED":
                    self.logger.warning("GitHub API rate limit exceeded")

                    # Get rate limit info if available
                    rate_limit_info = {}
                    if "headers" in response and "x-ratelimit-reset" in response["headers"]:
                        reset_time = int(response["headers"]["x-ratelimit-reset"])
                        current_time = int(time.time())
                        wait_time = max(0, reset_time - current_time)

                        rate_limit_info = {
                            "limit": response["headers"].get("x-ratelimit-limit", "unknown"),
                            "remaining": response["headers"].get("x-ratelimit-remaining", "0"),
                            "reset_time": datetime.fromtimestamp(reset_time).strftime("%Y-%m-%d %H:%M:%S"),
                            "wait_seconds": wait_time
                        }

                    return True, rate_limit_info

        return False, {}

    def _execute_with_rate_limit_handling(self, action_func, *args, **kwargs):
        """
        Execute a function with rate limit handling.

        Parameters:
            action_func (Callable): The function to execute
            *args, **kwargs: Arguments to pass to the function

        Returns:
            ActionResult: The result of the action
        """
        try:
            result = action_func(*args, **kwargs)

            # Check if this is a rate limit error
            if not result.success and "rate limit" in result.message.lower():
                self.logger.warning(f"Rate limit detected: {result.message}")

                # Add helpful information to the message
                result.message = f"GitHub API rate limit exceeded. Please try again later. {result.message}"

            return result

        except Exception as e:
            self.logger.error(f"Error executing action: {str(e)}")
            return ActionResult(
                success=False,
                data={},
                message=f"Action failed: {str(e)}"
            )

    def _validate_parameters(self, action: Action) -> Tuple[bool, List, str]:
        """
        Validate that all required parameters for an action are present.

        Parameters:
            action (Action): The action to validate

        Returns:
            Tuple[bool, List, str]: (is_valid, missing_params, message)
        """

        if action.type not in REQUIRED_PARAMETERS:
            return False, [], f"Unknown action type: {action.type.value}"

        provided_params = set(action.parameters.keys())
        required_param_set = set(REQUIRED_PARAMETERS[action.type])
        missing_params = required_param_set - provided_params

        if not missing_params:
            return True, [], "All required parameters are present"

        # Generate a helpful message about missing parameters
        message = f"Missing required parameters for {action.type.value}:\n"
        for param in missing_params:
            description = PARAMETER_DESCRIPTIONS.get(param, param)
            message += f"- {param}: {description}\n"

        return False, list(missing_params), message

    def _generate_user_friendly_error(self, action_type: ActionType, error_message: str) -> str:
        """
        Generate a user-friendly error message based on the action type and error.

        Parameters:
            action_type (ActionType): The type of action that failed
            error_message (str): The technical error message

        Returns:
            str: A user-friendly error message
        """
        # Base error message with technical details hidden unless needed
        base_message = ERROR_MESSAGES["BASE"]

        # Customize based on action type
        if action_type in ERROR_MESSAGES:
            friendly_message = base_message + ERROR_MESSAGES[action_type]
        else:
            friendly_message = base_message + f"complete the {action_type.value.lower().replace('_', ' ')} action."

        # Get suggestions based on action type
        if action_type in ERROR_MESSAGES:
            suggestions = ERROR_MESSAGES.get(f"{action_type.value}_SUGGESTIONS",
                                             ERROR_MESSAGES["DEFAULT_SUGGESTIONS"])
        else:
            suggestions = ERROR_MESSAGES["DEFAULT_SUGGESTIONS"]

        # Add technical details for debugging
        technical_details = f"\n\nTechnical details (for troubleshooting): {error_message}"

        # Format the full message
        full_message = f"{friendly_message}\n\nHere are some suggestions:\n"
        for suggestion in suggestions:
            full_message += f"- {suggestion}\n"

        full_message += technical_details

        return full_message

    def _enhance_success_message(self, action_type: ActionType, action_result: ActionResult) -> str:
        """
        Enhance the success message to make it more informative for the user.

        Parameters:
            action_type (ActionType): The type of action that succeeded
            action_result (ActionResult): The result of the action

        Returns:
            str: An enhanced success message
        """
        base_message = SUCCESS_MESSAGES["BASE"]

        # Customize based on action type
        if action_type == ActionType.GET_PROJECT_INFO:
            project_title = action_result.data.get("title", "the project")
            project_owner = action_result.data.get("owner", "the owner")
            return f"{base_message}{SUCCESS_MESSAGES['GET_PROJECT_INFO'].format(title=project_title, owner=project_owner)}"

        elif action_type == ActionType.GET_USER_PROJECTS:
            username = action_result.data.get("projects", [])[0].get("owner", "the user") if action_result.data.get("projects") else "the user"
            project_count = len(action_result.data.get("projects", []))
            return f"{base_message}{SUCCESS_MESSAGES['GET_USER_PROJECTS'].format(count=project_count, username=username)}"

        elif action_type == ActionType.GET_ORG_PROJECTS:
            org_name = action_result.data.get("projects", [])[0].get("owner", "the organization") if action_result.data.get("projects") else "the organization"
            project_count = len(action_result.data.get("projects", []))
            return f"{base_message}{SUCCESS_MESSAGES['GET_ORG_PROJECTS'].format(count=project_count, org_name=org_name)}"

        elif action_type == ActionType.CONVERT_DRAFT_TO_ISSUE:
            repo = action_result.data.get("repo", TEST_REPOSITORY)
            return f"{base_message}{SUCCESS_MESSAGES['CONVERT_DRAFT_TO_ISSUE'].format(repo=repo)}"

        elif action_type == ActionType.ADD_COMMENT_TO_ISSUE:
            repo = action_result.data.get("repo", TEST_REPOSITORY)
            issue_number = action_result.data.get("issue_number", "the issue")
            return f"{base_message}{SUCCESS_MESSAGES['ADD_COMMENT_TO_ISSUE'].format(repo=repo, issue_number=issue_number)}"

        elif action_type == ActionType.CREATE_ISSUE:
            repo = action_result.data.get("repo", TEST_REPOSITORY)
            issue_number = action_result.data.get("issue_number", "the new issue")
            return f"{base_message}{SUCCESS_MESSAGES['CREATE_ISSUE'].format(repo=repo, issue_number=issue_number)}"

        elif action_type == ActionType.GET_REPOSITORY_ID:
            repo = action_result.data.get("name", TEST_REPOSITORY)
            return f"{base_message}{SUCCESS_MESSAGES['GET_REPOSITORY_ID'].format(repo=repo)}"

        elif action_type in SUCCESS_MESSAGES:
            return f"{base_message}{SUCCESS_MESSAGES[action_type]}"

        else:
            return f"{base_message}completed the {action_type.value.lower().replace('_', ' ')} action."

    def _refresh_rate_limit_info(self):
        """Refresh rate limit information from GitHub API."""
        try:
            # Simple request to check rate limit
            query = """
            query {
                rateLimit {
                    limit
                    remaining
                    resetAt
                }
            }
            """

            response = self.gh_client.execute_graphql(query)

            if "data" in response and "rateLimit" in response["data"]:
                rate_limit = response["data"]["rateLimit"]

                self.rate_limit_info["limit"] = rate_limit.get("limit", API_RATE_LIMIT)
                self.rate_limit_info["remaining"] = rate_limit.get("remaining", API_RATE_LIMIT)

                # Parse the reset time string to a datetime object
                reset_at = rate_limit.get("resetAt")
                if reset_at:
                    try:
                        # GitHub returns ISO format
                        reset_time = datetime.fromisoformat(reset_at.replace('Z', '+00:00'))
                        self.rate_limit_info["reset_time"] = reset_time
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not parse rate limit reset time: {reset_at}")

                self.rate_limit_info["last_updated"] = time.time()

                self.logger.info(f"Updated rate limit info: "
                                 f"limit={self.rate_limit_info['limit']}, "
                                 f"remaining={self.rate_limit_info['remaining']}, "
                                 f"reset_time={self.rate_limit_info['reset_time']}")

        except Exception as e:
            self.logger.error(f"Failed to refresh rate limit info: {str(e)}")
