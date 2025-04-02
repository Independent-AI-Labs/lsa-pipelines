import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from integration.pipelines.pipelines.github_pm_pipeline_impl.data.pm_models import ActionContext

from integration.github.github_client import GitHubClient
from integration.github.models.project_models import Action, ActionType, ActionResult
from integration.pipelines.pipelines.github_pm_pipeline_impl.data.pm_constants import (
    GITHUB_TOKEN,
    BASE_API_URL,
    API_RATE_LIMIT,
    DEFAULT_LABELS,
    TEST_REPOSITORY
)
from integration.pipelines.pipelines.github_pm_pipeline_impl.data.prompts import (
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    PARAMETER_DESCRIPTIONS,
    REQUIRED_PARAMETERS
)


class ActionHandler:
    """
    Handles execution of GitHub API actions and manages context.

    This class is responsible for:
    1. Validating action parameters
    2. Executing actions against the GitHub API
    3. Managing and updating context information
    4. Resolving missing parameters when possible
    5. Handling errors and rate limiting
    """

    def __init__(self):
        """Initialize the ActionHandler."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize GitHub client
        self.gh_client = GitHubClient(GITHUB_TOKEN, BASE_API_URL)

        # Track rate limit information
        self.rate_limit_info = {
            "limit": API_RATE_LIMIT,
            "remaining": API_RATE_LIMIT,
            "reset_time": None,
            "last_updated": time.time()
        }

    def prepare_action_parameters(self, action: Action, context: ActionContext) -> Tuple[Action, ActionContext]:
        """
        Validate and retrieve required IDs before executing the action.

        Args:
            action: The action to prepare
            context: The current context

        Returns:
            Tuple[Action, ActionContext]: The updated action and context
        """
        # Don't need to prepare parameters for REPLY_TO_USER
        if action.type == ActionType.REPLY_TO_USER:
            return action, context

        # Copy the action parameters to avoid modifying the original
        updated_params = action.parameters.copy()

        # Handle entity (user/org) resolution
        if action.type in [ActionType.GET_USER_PROJECTS, ActionType.GET_ORG_PROJECTS, ActionType.GET_PROJECT_INFO]:
            # Check if we need to resolve entity name
            if action.type == ActionType.GET_USER_PROJECTS and "username" not in updated_params and context.entity_name:
                updated_params["username"] = context.entity_name

            if action.type == ActionType.GET_ORG_PROJECTS and "org_name" not in updated_params and context.entity_name:
                updated_params["org_name"] = context.entity_name

            if action.type == ActionType.GET_PROJECT_INFO:
                if "owner" not in updated_params and context.entity_name:
                    updated_params["owner"] = context.entity_name
                if "project_number" not in updated_params and context.project_number:
                    updated_params["project_number"] = context.project_number

        # Handle project ID resolution
        project_id_needed = any(param in ["project_id", "PROJECT_ID", "%PROJECT_ID%"]
                                for param in updated_params.values())

        if project_id_needed or action.type in [
            ActionType.GET_PROJECT_FIELDS,
            ActionType.GET_PROJECT_ITEMS,
            ActionType.ADD_ITEM_TO_PROJECT,
            ActionType.ADD_DRAFT_ISSUE,
            ActionType.UPDATE_PROJECT_SETTINGS,
            ActionType.UPDATE_TEXT_FIELD,
            ActionType.UPDATE_SELECT_FIELD,
            ActionType.UPDATE_ITERATION_FIELD,
            ActionType.DELETE_PROJECT_ITEM,
            ActionType.CONVERT_DRAFT_TO_ISSUE
        ]:
            # If we have project_id in context but not in parameters, add it
            if context.project_id and "project_id" not in updated_params:
                updated_params["project_id"] = context.project_id
            # If we have project_id in parameters as a placeholder, replace it
            elif "project_id" in updated_params and updated_params["project_id"] in [
                "GET_FROM_INFO", "PROJECT_ID", "%PROJECT_ID%"
            ] and context.project_id:
                updated_params["project_id"] = context.project_id
            # If we still don't have project_id but have owner and project_number, get it
            elif (not context.project_id or "project_id" not in updated_params) and context.entity_name and context.project_number:
                # Fetch the project info to get the ID
                project_info_result = self.gh_client.get_project_info(
                    owner=context.entity_name,
                    project_number=context.project_number
                )

                if project_info_result.success:
                    context.project_id = project_info_result.data.get("id")
                    context.project_title = project_info_result.data.get("title")
                    updated_params["project_id"] = context.project_id

        # Handle item ID resolution
        item_id_needed = any(param in ["item_id", "ITEM_ID", "%ITEM_ID%", "draft_item_id"]
                             for param in updated_params.values())

        if item_id_needed or action.type in [
            ActionType.UPDATE_TEXT_FIELD,
            ActionType.UPDATE_SELECT_FIELD,
            ActionType.UPDATE_ITERATION_FIELD,
            ActionType.DELETE_PROJECT_ITEM,
            ActionType.CONVERT_DRAFT_TO_ISSUE
        ]:
            # If we have item_id in context but not in parameters, add it
            if context.item_id:
                if "item_id" not in updated_params:
                    updated_params["item_id"] = context.item_id
                if "draft_item_id" not in updated_params and action.type == ActionType.CONVERT_DRAFT_TO_ISSUE:
                    updated_params["draft_item_id"] = context.item_id
            # If we have item_id in parameters as a placeholder, replace it
            elif "item_id" in updated_params and updated_params["item_id"] in [
                "GET_FROM_ITEM_INFO", "ITEM_ID", "%ITEM_ID%"
            ] and context.item_id:
                updated_params["item_id"] = context.item_id
            # Same for draft_item_id
            elif "draft_item_id" in updated_params and updated_params["draft_item_id"] in [
                "GET_FROM_ITEM_INFO", "ITEM_ID", "%ITEM_ID%"
            ] and context.item_id:
                updated_params["draft_item_id"] = context.item_id

        # Handle field ID resolution
        field_id_needed = any(param in ["field_id", "FIELD_ID", "%FIELD_ID%"]
                              for param in updated_params.values())

        if field_id_needed or action.type in [
            ActionType.UPDATE_TEXT_FIELD,
            ActionType.UPDATE_SELECT_FIELD,
            ActionType.UPDATE_ITERATION_FIELD
        ]:
            # If we have field_id in context but not in parameters, add it
            if context.field_id and "field_id" not in updated_params:
                updated_params["field_id"] = context.field_id
            # If we have field_id in parameters as a placeholder, replace it
            elif "field_id" in updated_params and updated_params["field_id"] in [
                "GET_FROM_FIELD_INFO", "FIELD_ID", "%FIELD_ID%"
            ] and context.field_id:
                updated_params["field_id"] = context.field_id
            # If we have field_name but not field_id, try to fetch it
            elif context.field_name and not context.field_id and context.project_id:
                # Fetch the project fields to get the ID
                fields_result = self.gh_client.get_project_fields(
                    project_id=context.project_id
                )

                if fields_result.success:
                    fields = fields_result.data.get("fields", [])
                    for field in fields:
                        if field.get("name").lower() == context.field_name.lower():
                            context.field_id = field.get("id")
                            context.field_type = field.get("type")
                            context.field_options = field.get("options")
                            updated_params["field_id"] = context.field_id

                            # Update the project custom fields in context
                            self._update_project_custom_fields(context, fields_result.data.get("fields", []))
                            break

        # Handle option ID resolution for select fields
        option_id_needed = any(param in ["option_id", "OPTION_ID", "%OPTION_ID%"]
                               for param in updated_params.values())

        if option_id_needed or action.type == ActionType.UPDATE_SELECT_FIELD:
            # If we have selected_option_id in context but not in parameters, add it
            if context.selected_option_id and "option_id" not in updated_params:
                updated_params["option_id"] = context.selected_option_id
            # If we have option_id in parameters as a placeholder, replace it
            elif "option_id" in updated_params and updated_params["option_id"] in [
                "ID", "OPTION_ID", "%OPTION_ID%"
            ] and context.selected_option_id:
                updated_params["option_id"] = context.selected_option_id
            # If we have selected_option_name but not selected_option_id, try to resolve it
            elif (context.selected_option_name and not context.selected_option_id and
                  context.field_options and context.field_type == "single_select"):
                # Look up the option ID from the name
                for option_id, option_name in context.field_options.items():
                    if option_name.lower() == context.selected_option_name.lower():
                        context.selected_option_id = option_id
                        updated_params["option_id"] = option_id
                        break

                # If we still don't have an option ID, try to look it up in custom fields
                if not context.selected_option_id and context.field_name and context.custom_fields:
                    field_found = False
                    for field in context.custom_fields:
                        if field.get("name").lower() == context.field_name.lower():
                            field_found = True
                            option_id = self._find_option_id_in_field(field, context.selected_option_name)
                            if option_id:
                                context.selected_option_id = option_id
                                updated_params["option_id"] = option_id
                                break

                    # If field not found, try to get fields and retry
                    if not field_found and context.project_id and not "field_name" in updated_params:
                        fields_result = self.gh_client.get_project_fields(
                            project_id=context.project_id
                        )

                        if fields_result.success:
                            self._update_project_custom_fields(context, fields_result.data.get("fields", []))
                            # Try to find the option ID again
                            for field in context.custom_fields:
                                if field.get("name").lower() == context.field_name.lower():
                                    option_id = self._find_option_id_in_field(field, context.selected_option_name)
                                    if option_id:
                                        context.selected_option_id = option_id
                                        updated_params["option_id"] = option_id
                                        break

        # Handle iteration ID resolution
        iteration_id_needed = any(param in ["iteration_id", "ITERATION_ID", "%ITERATION_ID%"]
                                  for param in updated_params.values())

        if iteration_id_needed or action.type == ActionType.UPDATE_ITERATION_FIELD:
            # If we have selected_iteration_id in context but not in parameters, add it
            if context.selected_iteration_id and "iteration_id" not in updated_params:
                updated_params["iteration_id"] = context.selected_iteration_id
            # If we have iteration_id in parameters as a placeholder, replace it
            elif "iteration_id" in updated_params and updated_params["iteration_id"] in [
                "ID", "ITERATION_ID", "%ITERATION_ID%"
            ] and context.selected_iteration_id:
                updated_params["iteration_id"] = context.selected_iteration_id

        # Handle repository resolution
        repo_needed = any(param in ["repo", "REPO", "%REPO%", "repository"]
                          for param in updated_params.values())

        if repo_needed or action.type in [
            ActionType.CONVERT_DRAFT_TO_ISSUE,
            ActionType.ADD_COMMENT_TO_ISSUE,
            ActionType.CREATE_ISSUE,
            ActionType.GET_REPOSITORY_ID
        ]:
            # If we have repo_name in context but not in parameters, add it
            if context.repo_name and "repo" not in updated_params:
                updated_params["repo"] = context.repo_name
            # If we have repo in parameters as a placeholder, replace it
            elif "repo" in updated_params and updated_params["repo"] in [
                "REPO", "%REPO%", "repository"
            ] and context.repo_name:
                updated_params["repo"] = context.repo_name

            # Handle repo owner similarly
            if context.repo_owner and "owner" not in updated_params:
                updated_params["owner"] = context.repo_owner
            elif "owner" in updated_params and updated_params["owner"] in [
                "OWNER", "%OWNER%"
            ] and context.repo_owner:
                updated_params["owner"] = context.repo_owner

        # Create updated action with resolved parameters
        updated_action = Action(
            type=action.type,
            parameters=updated_params
        )

        self.logger.info(f"Prepared action parameters: {updated_action.parameters}")

        return updated_action, context

    def _find_option_id_in_field(self, field: Dict, option_name: str) -> Optional[str]:
        """
        Find an option ID in a field based on the option name.

        Args:
            field: The field dictionary
            option_name: The name of the option to find

        Returns:
            Optional[str]: The option ID if found, None otherwise
        """
        if not field or not option_name:
            return None

        # Handle different field structures
        if "options" in field:
            # Direct options mapping
            for option_id, name in field.get("options", {}).items():
                if name.lower() == option_name.lower():
                    return option_id
        elif "configuration" in field:
            # Some fields have configurations with options
            options = field.get("configuration", {}).get("options", [])
            for option in options:
                if option.get("name", "").lower() == option_name.lower():
                    return option.get("id")

        return None

    def _update_project_custom_fields(self, context: ActionContext, fields: List[Dict]):
        """
        Update the context with project custom fields.

        Args:
            context: The action context to update
            fields: List of fields from the API
        """
        if not fields:
            return

        context.custom_fields = fields

        # Extract issue types, statuses, and categories
        issue_types = []
        statuses = []
        categories = []
        labels = []

        for field in fields:
            field_name = field.get("name", "").lower()

            # Process based on field name and type
            if field_name in ["type", "issue type", "item type"] and field.get("type") == "single_select":
                issue_types = self._extract_options_from_field(field)
                context.issue_types = issue_types

            elif field_name in ["status", "state"] and field.get("type") == "single_select":
                statuses = self._extract_options_from_field(field)
                context.statuses = statuses

            elif field_name in ["category", "area", "component"] and field.get("type") == "single_select":
                categories = self._extract_options_from_field(field)
                context.categories = categories

            elif field_name in ["label", "labels", "tag", "tags"] and field.get("type") == "single_select":
                labels = self._extract_options_from_field(field)
                context.labels = labels

    def _extract_options_from_field(self, field: Dict) -> List[str]:
        """
        Extract options from a field.

        Args:
            field: The field dictionary

        Returns:
            List[str]: List of option names
        """
        options = []

        # Handle different field structures
        if "options" in field:
            # Direct options mapping
            options = list(field.get("options", {}).values())
        elif "configuration" in field:
            # Some fields have configurations with options
            config_options = field.get("configuration", {}).get("options", [])
            options = [opt.get("name") for opt in config_options if "name" in opt]

        return options

    def execute_action(self, action: Action) -> ActionResult:
        """
        Execute a GitHub API action and return the result.

        Parameters:
            action (Action): The action to execute

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
                # This action doesn't need GitHub API calls
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

    def update_context_from_result(self, action: Action, result: ActionResult, context: ActionContext) -> ActionContext:
        """
        Update the context with information from the action result.

        Args:
            action: The executed action
            result: The result of the action
            context: The current context

        Returns:
            ActionContext: The updated context
        """
        if not result.success:
            return context

        # Create a copy of the context to update
        updated_context = ActionContext(**context.dict())

        # Update context based on action type and result
        if action.type == ActionType.GET_PROJECT_INFO:
            updated_context.project_id = result.data.get("id")
            updated_context.project_title = result.data.get("title")
            updated_context.entity_name = result.data.get("owner")
            updated_context.project_number = result.data.get("number")

        elif action.type == ActionType.GET_USER_PROJECTS:
            updated_context.entity_type = "User"
            updated_context.entity_name = action.parameters.get("username")

        elif action.type == ActionType.GET_ORG_PROJECTS:
            updated_context.entity_type = "Organization"
            updated_context.entity_name = action.parameters.get("org_name")

        elif action.type == ActionType.GET_PROJECT_FIELDS:
            # Store field information if a specific field was requested
            field_name = action.parameters.get("field_name")

            # Update custom fields regardless of specific field request
            self._update_project_custom_fields(updated_context, result.data.get("fields", []))

            if field_name:
                fields = result.data.get("fields", [])
                for field in fields:
                    if field.get("name").lower() == field_name.lower():
                        updated_context.field_id = field.get("id")
                        updated_context.field_name = field.get("name")
                        updated_context.field_type = field.get("type")
                        updated_context.field_options = field.get("options")
                        updated_context.iterations = field.get("iterations")
                        break

        elif action.type == ActionType.GET_PROJECT_ITEMS:
            # If a specific item was requested by title, store its ID
            item_title = action.parameters.get("item_title")
            if item_title:
                items = result.data.get("items", [])
                for item in items:
                    if item.get("title").lower() == item_title.lower():
                        updated_context.item_id = item.get("id")
                        updated_context.item_title = item.get("title")
                        break

        elif action.type == ActionType.ADD_DRAFT_ISSUE:
            updated_context.item_id = result.data.get("item_id")
            updated_context.item_title = action.parameters.get("title")

        elif action.type == ActionType.CONVERT_DRAFT_TO_ISSUE:
            updated_context.item_id = result.data.get("project_item_id")

        elif action.type == ActionType.GET_REPOSITORY_ID:
            updated_context.repo_id = result.data.get("id")
            updated_context.repo_name = result.data.get("name")
            updated_context.repo_owner = action.parameters.get("owner")

        return updated_context

    def validate_parameters(self, action: Action) -> Tuple[bool, List, str]:
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

    def requires_user_input(self, missing_params: List[str], context: ActionContext) -> bool:
        """
        Determine if we need user input to resolve missing parameters.

        Args:
            missing_params: List of missing parameter names
            context: The current context

        Returns:
            bool: True if user input is required
        """
        # Check if any of the missing parameters can't be resolved from context or API calls
        critical_params = {
            "owner": not bool(context.entity_name),
            "username": not bool(context.entity_name),
            "org_name": not bool(context.entity_name),
            "project_number": not bool(context.project_number) and not bool(context.project_title),
            "title": True,  # Always needs user input
            "body": True,  # Always needs user input
            "repo": not bool(context.repo_name)
        }

        for param in missing_params:
            if param in critical_params and critical_params[param]:
                return True

        return False

    def resolve_missing_parameters(self, missing_params: List[str],
                                   action: Action, context: ActionContext) -> Tuple[Dict[str, Any], ActionContext]:
        """
        Try to resolve missing parameters through API calls.

        Args:
            missing_params: List of missing parameter names
            action: The current action
            context: The current context

        Returns:
            Tuple[Dict[str, Any], ActionContext]: Resolved parameters and updated context
        """
        resolved_params = {}
        updated_context = ActionContext(**context.dict())

        # Try to resolve each missing parameter
        for param in missing_params:
            # Project ID resolution
            if param == "project_id" and updated_context.entity_name and updated_context.project_number:
                # Try to get project info
                project_info_result = self.gh_client.get_project_info(
                    owner=updated_context.entity_name,
                    project_number=updated_context.project_number
                )

                if project_info_result.success:
                    resolved_params["project_id"] = project_info_result.data.get("id")
                    updated_context.project_id = project_info_result.data.get("id")
                    updated_context.project_title = project_info_result.data.get("title")

            # Field ID resolution
            elif param == "field_id" and updated_context.field_name and updated_context.project_id:
                # Try to get project fields
                fields_result = self.gh_client.get_project_fields(
                    project_id=updated_context.project_id
                )

                if fields_result.success:
                    fields = fields_result.data.get("fields", [])

                    # Update custom fields in context
                    self._update_project_custom_fields(updated_context, fields)

                    for field in fields:
                        if field.get("name").lower() == updated_context.field_name.lower():
                            resolved_params["field_id"] = field.get("id")
                            updated_context.field_id = field.get("id")
                            updated_context.field_type = field.get("type")
                            updated_context.field_options = field.get("options")
                            break

            # Item ID resolution
            elif param == "item_id" and updated_context.item_title and updated_context.project_id:
                # Try to get project items
                items_result = self.gh_client.get_project_items(
                    project_id=updated_context.project_id
                )

                if items_result.success:
                    items = items_result.data.get("items", [])
                    for item in items:
                        if item.get("title").lower() == updated_context.item_title.lower():
                            resolved_params["item_id"] = item.get("id")
                            updated_context.item_id = item.get("id")
                            break

            # Option ID resolution
            elif param == "option_id" and updated_context.selected_option_name and updated_context.field_options:
                # Look up option ID from name
                for option_id, option_name in updated_context.field_options.items():
                    if option_name.lower() == updated_context.selected_option_name.lower():
                        resolved_params["option_id"] = option_id
                        updated_context.selected_option_id = option_id
                        break

            # Repository ID resolution
            elif param == "repo_id" and updated_context.repo_owner and updated_context.repo_name:
                # Try to get repository ID
                repo_result = self.gh_client.get_repository_id(
                    owner=updated_context.repo_owner,
                    repo=updated_context.repo_name
                )

                if repo_result.success:
                    resolved_params["repo_id"] = repo_result.data.get("id")
                    updated_context.repo_id = repo_result.data.get("id")

        return resolved_params, updated_context

    def should_provide_intermediate_response(self, action: Action, result: ActionResult) -> bool:
        """
        Determine if we should provide an intermediate response to the user.

        Args:
            action: The executed action
            result: The result of the action

        Returns:
            bool: True if we should provide an intermediate response
        """
        # If the action succeeded, we generally don't need an intermediate response
        if result.success:
            return False

        # For certain failures, we might want to ask the user for more information
        if "not found" in result.message.lower():
            return True

        if "permission" in result.message.lower() or "unauthorized" in result.message.lower():
            return True

        if "invalid" in result.message.lower() and any(param in result.message.lower()
                                                       for param in ["id", "number", "name"]):
            return True

        # For parameter validation failures
        if "missing" in result.message.lower() and "parameter" in result.message.lower():
            return True

        return False

    def generate_user_friendly_error(self, action_type: ActionType, error_message: str) -> str:
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

    def enhance_success_message(self, action_type: ActionType, action_result: ActionResult) -> str:
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

    def refresh_rate_limit_info(self):
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
