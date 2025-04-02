import json
import logging
import re
from typing import Any, Dict, Generator, Iterator, List, Union, Tuple, Optional

from integration.pipelines.pipelines.github_pm_pipeline_impl.data.pm_models import ActionContext, ChatInfo

from integration.github.models.project_models import Action, ActionType, ActionResult
from integration.net.ollama.ollama_api import prompt_model
from integration.pipelines.pipelines.github_pm_pipeline_impl.action_handler import ActionHandler
# Import constants and prompts
from integration.pipelines.pipelines.github_pm_pipeline_impl.data.pm_constants import (
    DEFAULT_LLM,
    MAX_ACTIONS,
    LOG_LEVEL,
    LOG_FORMAT
)
from integration.pipelines.pipelines.github_pm_pipeline_impl.data.prompts import (
    SYSTEM_PROMPT,
    ACTION_DECISION_PROMPT,
    RESULT_ANALYSIS_PROMPT,
    USER_REPLY_PROMPT,
    CONTEXT_ANALYSIS_PROMPT
)


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

        # Initialize action handler
        self.action_handler = ActionHandler()

        # Model configuration
        self.default_model = DEFAULT_LLM

    async def on_startup(self):
        """Initialize resources when the server starts."""
        self.logger.info(f"on_startup:{__name__}")
        # Initialize rate limit info
        self.action_handler.refresh_rate_limit_info()

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
        yield "Analyzing user message and conversation context..."

        # Store the pipeline model_id for dedicated chats and use default_model for LLM operations
        pipeline_model_id = model_id  # This is the model that represents the pipeline itself
        llm = self.default_model  # This is what we'll use for actual LLM operations

        try:
            # Deep analysis of conversation to identify entities and context
            context = self._analyze_conversation_context(user_message, messages, llm)

            # Log the identified context
            context_str = self._format_context_for_logging(context)
            yield f"\nIdentified context:\n{context_str}"

            # Process conversation history to decide on action
            action = self._decide_action(user_message, messages, llm, context)
            yield f"\nAction decided: {action.type.value}"

            # Validate and retrieve required IDs before executing action
            action, context = self.action_handler.prepare_action_parameters(action, context)
            yield f"\nAction parameters prepared"

            # Execute action chain until we have a final response
            final_response = self._execute_action_chain(action, context, chat_id, user_id, messages, llm)

            # Return the response
            yield "</think>"
            yield final_response

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            error_response = f"I encountered an error while processing your request: {str(e)}"
            yield "</think>"
            yield error_response

    def _format_context_for_logging(self, context: ActionContext) -> str:
        """Format context for logging purposes."""
        output = []

        def add_if_exists(name, value):
            if value is not None:
                output.append(f"{name}: {value}")

        # Entity info
        add_if_exists("Entity name", context.entity_name)
        add_if_exists("Entity type", context.entity_type)

        # Project info
        add_if_exists("Project ID", context.project_id)
        add_if_exists("Project number", context.project_number)
        add_if_exists("Project title", context.project_title)

        # Repository info
        add_if_exists("Repository", context.repo_name)
        add_if_exists("Repository owner", context.repo_owner)

        # Field info
        add_if_exists("Field name", context.field_name)
        add_if_exists("Field type", context.field_type)
        add_if_exists("Selected option", context.selected_option_name)

        # Custom fields
        if context.custom_fields:
            output.append("Custom fields:")
            for field in context.custom_fields:
                output.append(f"  - {field.get('name', 'Unknown')} ({field.get('type', 'Unknown')})")

        # Special field types
        if context.issue_types:
            output.append(f"Issue types: {', '.join(context.issue_types)}")
        if context.statuses:
            output.append(f"Statuses: {', '.join(context.statuses)}")
        if context.categories:
            output.append(f"Categories: {', '.join(context.categories)}")

        return "\n".join(output)

    def _extract_chat_info(self, body: Dict[str, Any]) -> ChatInfo:
        """Extract chat information from the request body."""
        return ChatInfo(
            chat_id=body.get("chat_id", "unknown"),
            user_id=body.get("user_id", "unknown")
        )

    def _analyze_conversation_context(self, user_message: str, messages: List[Dict], model: str) -> ActionContext:
        """
        Analyze the conversation history to identify entities, projects, and other context.

        Args:
            user_message: The current user message
            messages: The conversation history
            model: The LLM to use for analysis

        Returns:
            ActionContext: The identified context
        """
        # Format conversation history for the LLM
        formatted_history = self._format_conversation_history(messages)

        # Prepare prompt for context analysis
        prompt = CONTEXT_ANALYSIS_PROMPT.format(
            conversation_history=formatted_history,
            current_message=user_message
        )

        # Get LLM analysis
        response = prompt_model(prompt, model=model, system_prompt=SYSTEM_PROMPT)

        # Initialize context
        context = ActionContext()

        # Extract information using regex patterns
        self._extract_context_with_regex(response, context)

        # Also check the conversation history for additional context
        self._extract_context_from_history(messages, context)

        return context

    def _extract_context_with_regex(self, response: str, context: ActionContext):
        """
        Extract context information using regex patterns.

        Args:
            response: The LLM response
            context: The context to update
        """
        # Extract entity name (user or org)
        entity_match = re.search(r'Entity(?:\s+name)?(?:\s*:)?\s+([a-zA-Z0-9-]+)', response, re.IGNORECASE)
        if entity_match:
            context.entity_name = entity_match.group(1)

        # Extract entity type
        entity_type_match = re.search(r'Entity(?:\s+type)?(?:\s*:)?\s+([a-zA-Z0-9-]+)', response, re.IGNORECASE)
        if entity_type_match:
            entity_type = entity_type_match.group(1)
            if entity_type.lower() in ["organization", "org"]:
                context.entity_type = "Organization"
            elif entity_type.lower() in ["user", "individual"]:
                context.entity_type = "User"

        # Extract project info
        project_id_match = re.search(r'Project(?:\s+ID)?(?:\s*:)?\s+([a-zA-Z0-9_]+)', response, re.IGNORECASE)
        if project_id_match:
            context.project_id = project_id_match.group(1)

        project_num_match = re.search(r'Project(?:\s+number)?(?:\s*:)?\s+(\d+)', response, re.IGNORECASE)
        if project_num_match:
            try:
                context.project_number = int(project_num_match.group(1))
            except ValueError:
                pass

        project_title_match = re.search(r'Project(?:\s+title)?(?:\s*:)?\s+(.+?)(?:\n|$)', response, re.IGNORECASE)
        if project_title_match:
            context.project_title = project_title_match.group(1).strip()

        # Extract repository info
        repo_match = re.search(r'Repository(?:\s*:)?\s+([a-zA-Z0-9-]+)', response, re.IGNORECASE)
        if repo_match:
            context.repo_name = repo_match.group(1)

        repo_owner_match = re.search(r'Repository(?:\s+owner)?(?:\s*:)?\s+([a-zA-Z0-9-]+)', response, re.IGNORECASE)
        if repo_owner_match:
            context.repo_owner = repo_owner_match.group(1)

        # Extract field info
        field_match = re.search(r'Field(?:\s+name)?(?:\s*:)?\s+([a-zA-Z0-9 ]+)', response, re.IGNORECASE)
        if field_match:
            context.field_name = field_match.group(1).strip()

        option_match = re.search(r'(?:Selected\s+)?Option(?:\s+name)?(?:\s*:)?\s+([a-zA-Z0-9 ]+)', response, re.IGNORECASE)
        if option_match:
            context.selected_option_name = option_match.group(1).strip()

        # Extract issue types, statuses, categories
        for field_type in ["Issue types", "Statuses", "Categories", "Labels"]:
            pattern = rf'{field_type}(?:\s*:)?\s+(.+?)(?:\n|$)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                values = [v.strip() for v in match.group(1).split(',')]
                if field_type.lower() == "issue types":
                    context.issue_types = values
                elif field_type.lower() == "statuses":
                    context.statuses = values
                elif field_type.lower() == "categories":
                    context.categories = values
                elif field_type.lower() == "labels":
                    context.labels = values

    def _extract_context_from_history(self, messages: List[Dict], context: ActionContext):
        """
        Extract additional context from previous assistant messages that might contain API results.

        Args:
            messages: The conversation history
            context: The current context to update
        """
        # Only check assistant messages
        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")

            # Look for project IDs in the content
            project_id_match = re.search(r'Project ID[:\s]+([A-Za-z0-9_]+)', content)
            if project_id_match and not context.project_id:
                context.project_id = project_id_match.group(1)

            # Look for project numbers
            project_num_match = re.search(r'Number[:\s]+(\d+)', content)
            if project_num_match and not context.project_number:
                try:
                    context.project_number = int(project_num_match.group(1))
                except ValueError:
                    pass

            # Look for project titles
            project_title_match = re.search(r'Project Title[:\s]+([^\n|]+)', content)
            if project_title_match and not context.project_title:
                context.project_title = project_title_match.group(1).strip()

            # Look for item IDs
            item_id_match = re.search(r'Item ID[:\s]+([A-Za-z0-9_]+)', content)
            if item_id_match and not context.item_id:
                context.item_id = item_id_match.group(1)

            # Look for field IDs
            field_id_match = re.search(r'Field ID[:\s]+([A-Za-z0-9_]+)', content)
            if field_id_match and not context.field_id:
                context.field_id = field_id_match.group(1)

            # Look for field names
            field_name_match = re.search(r'Field Name[:\s]+([^\n|]+)', content)
            if field_name_match and not context.field_name:
                context.field_name = field_name_match.group(1).strip()

            # Look for option names
            option_name_match = re.search(r'Option[:\s]+([^\n|]+)', content)
            if option_name_match and not context.selected_option_name:
                context.selected_option_name = option_name_match.group(1).strip()

            # Look for organization or user names
            entity_match = re.search(r'(retrieved|found).*?(projects for|information about).*?(user|organization) ([A-Za-z0-9-]+)', content)
            if entity_match and not context.entity_name:
                context.entity_name = entity_match.group(4)
                context.entity_type = "Organization" if entity_match.group(3) == "organization" else "User"

            # Look for status, category, type values
            for field_type in ["Status", "Category", "Type"]:
                value_match = re.search(rf'{field_type}[:\s]+([^\n|]+)', content)
                if value_match:
                    value = value_match.group(1).strip()
                    if field_type.lower() == "status" and not context.selected_option_name:
                        context.field_name = "Status"
                        context.selected_option_name = value
                    elif field_type.lower() == "category":
                        context.field_name = "Category"
                        context.selected_option_name = value
                    elif field_type.lower() == "type":
                        context.field_name = "Type"
                        context.selected_option_name = value

    def _decide_action(self, user_message: str, messages: List[Dict], model: str, context: Optional[ActionContext] = None) -> Action:
        """
        Analyze the user message and conversation history to decide on an action.

        Args:
            user_message: The user's message
            messages: The conversation history
            model: The LLM to use
            context: Optional context information

        Returns:
            Action: The decided action to take
        """
        # Format conversation history for the LLM
        formatted_history = self._format_conversation_history(messages)

        # Format context information for the prompt
        context_info = self._format_context_for_decision(context)

        # Prepare prompt for action decision
        prompt = ACTION_DECISION_PROMPT.format(
            conversation_history=formatted_history,
            current_message=user_message,
            context_info=context_info
        )

        # Get LLM decision
        response = prompt_model(prompt, model=model, system_prompt=SYSTEM_PROMPT)

        # Parse the response to get the decided action
        # The response should be in the format: ►ACTION_TYPE◄ followed by parameters
        action_type, params = self._parse_action_response(response)

        return Action(
            type=action_type,
            parameters=params
        )

    def _format_context_for_decision(self, context: Optional[ActionContext]) -> str:
        """Format context for action decision prompt."""
        if not context:
            return "No context information available."

        lines = ["Context Information:"]

        # Entity information
        if context.entity_name:
            lines.append(f"Entity: {context.entity_name}")
            if context.entity_type:
                lines.append(f"Entity type: {context.entity_type}")

        # Project information
        if context.project_id:
            lines.append(f"Project ID: {context.project_id}")
        if context.project_number:
            lines.append(f"Project number: {context.project_number}")
        if context.project_title:
            lines.append(f"Project title: {context.project_title}")

        # Repository information
        if context.repo_name:
            repo_str = context.repo_name
            if context.repo_owner:
                repo_str = f"{context.repo_owner}/{repo_str}"
            lines.append(f"Repository: {repo_str}")

        # Field information
        if context.field_name:
            lines.append(f"Field: {context.field_name}")
            if context.field_type:
                lines.append(f"Field type: {context.field_type}")
            if context.selected_option_name:
                lines.append(f"Selected option: {context.selected_option_name}")

        # Custom fields by type
        if context.issue_types:
            lines.append(f"Issue types: {', '.join(context.issue_types)}")
        if context.statuses:
            lines.append(f"Statuses: {', '.join(context.statuses)}")
        if context.categories:
            lines.append(f"Categories: {', '.join(context.categories)}")
        if context.labels:
            lines.append(f"Labels: {', '.join(context.labels)}")

        return "\n".join(lines)

    def _execute_action_chain(self, initial_action: Action, context: ActionContext,
                              chat_id: str, user_id: str, messages: List[Dict], model: str) -> str:
        """
        Execute a chain of actions until we have a final response to the user.

        Args:
            initial_action: The initial action to execute
            context: The current context
            chat_id: The current chat ID
            user_id: The current user ID
            messages: The conversation history
            model: The LLM to use

        Returns:
            str: The final response to send to the user
        """
        current_action = initial_action
        action_history = []
        current_context = context

        # Execute actions until we reach REPLY_TO_USER action or hit the safety limit
        while current_action.type != ActionType.REPLY_TO_USER and len(action_history) < MAX_ACTIONS:
            self.logger.info(f"Executing action {len(action_history) + 1}: {current_action.type.value}")

            # Validate parameters before executing
            is_valid, missing_params, validation_message = self.action_handler.validate_parameters(current_action)

            if not is_valid:
                # Generate intermediate response if parameters are missing and we need user input
                if self.action_handler.requires_user_input(missing_params, current_context):
                    # Generate a user-friendly message asking for the missing information
                    prompt = f"""
                    Based on the conversation so far, we need to ask the user for additional information.
                    The following parameters are missing: {missing_params}

                    Generate a friendly message asking the user to provide this information.
                    Make the message conversational and helpful, explaining why this information is needed.
                    """

                    intermediate_response = prompt_model(prompt, model=model, system_prompt=SYSTEM_PROMPT)

                    # Set next action to reply to user with the intermediate response
                    current_action = Action(
                        type=ActionType.REPLY_TO_USER,
                        parameters={"response": intermediate_response}
                    )
                    continue
                else:
                    # If parameters are invalid but we might be able to resolve them, add to history and continue
                    action_result = ActionResult(
                        success=False,
                        data={"missing_params": missing_params},
                        message=validation_message
                    )
                    action_history.append((current_action, action_result))

                    # Try to resolve missing parameters by making additional API calls
                    resolved_params, updated_context = self.action_handler.resolve_missing_parameters(
                        missing_params, current_action, current_context
                    )

                    if resolved_params:
                        # Update the action with resolved parameters
                        current_action = Action(
                            type=current_action.type,
                            parameters={**current_action.parameters, **resolved_params}
                        )
                        current_context = updated_context
                        continue
                    else:
                        # If we couldn't resolve parameters, set next action to reply to user with the error
                        current_action = Action(
                            type=ActionType.REPLY_TO_USER,
                            parameters={"error": validation_message}
                        )
                        continue

            # Execute the current action
            result = self.action_handler.execute_action(current_action)
            action_history.append((current_action, result))

            # Update context with results from the action
            current_context = self.action_handler.update_context_from_result(current_action, result, current_context)

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
                # Set next action to reply to user with the rate limit message
                current_action = Action(
                    type=ActionType.REPLY_TO_USER,
                    parameters={"error": "GitHub API rate limit exceeded. Please try again later."}
                )
                continue

            # Check if we need to evaluate intermediate response
            if not result.success and self.action_handler.should_provide_intermediate_response(current_action, result):
                # Generate an intermediate response for the user
                intermediate_prompt = f"""
                Based on the conversation so far and the following action result:

                Action: {current_action.type.value}
                Parameters: {json.dumps(current_action.parameters)}
                Result: {result.message}

                Generate a friendly message to the user explaining what went wrong,
                what is needed to proceed, and asking them for clarification or additional information.
                Make the message conversational and helpful.
                """

                intermediate_response = prompt_model(intermediate_prompt, model=model, system_prompt=SYSTEM_PROMPT)

                # Set next action to reply to user with the intermediate response
                current_action = Action(
                    type=ActionType.REPLY_TO_USER,
                    parameters={"response": intermediate_response}
                )
                continue

            # Analyze the result to decide on the next action
            next_action, should_continue = self._analyze_result(current_action, result, action_history,
                                                                current_context, messages, model)

            # If we should provide a direct response instead of continuing with actions
            if not should_continue:
                # Generate a response for this specific action result
                specific_response_prompt = f"""
                Generate a response to the user based on the following action result:

                Action: {current_action.type.value}
                Parameters: {json.dumps(current_action.parameters)}
                Result success: {result.success}
                Result message: {result.message}
                Result data: {json.dumps(result.data)}

                The response should be clear, concise, and directly address the user's request.
                """

                specific_response = prompt_model(specific_response_prompt, model=model, system_prompt=SYSTEM_PROMPT)

                return specific_response

            current_action = next_action

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
                return self.action_handler.generate_user_friendly_error(last_action.type, error_message)
            else:
                return f"I encountered an issue: {error_message}"

        # Check if we have a direct response parameter
        if current_action.type == ActionType.REPLY_TO_USER and "response" in current_action.parameters:
            return current_action.parameters["response"]

        # Generate user reply based on action history and context
        return self._generate_user_reply(action_history, current_context, messages, model)

    def _analyze_result(self, action: Action, result: ActionResult, action_history: List,
                        context: ActionContext, messages: List[Dict], model: str) -> Tuple[Action, bool]:
        """
        Analyze an action result to decide on the next action.

        Args:
            action: The current action
            result: The result of the action
            action_history: The history of actions and results
            context: The current context
            messages: The conversation history
            model: The LLM to use

        Returns:
            Tuple[Action, bool]: The next action and whether to continue action chain
        """
        # Format action history for the LLM
        formatted_history = self._format_action_history(action_history)

        # Format context information
        context_info = self._format_context_for_decision(context)

        # Format conversation history
        formatted_conv = self._format_conversation_history(messages)

        # Prepare prompt for result analysis
        prompt = RESULT_ANALYSIS_PROMPT.format(
            action_type=action.type.value,
            action_parameters=json.dumps(action.parameters, indent=2),
            result_success=result.success,
            result_data=json.dumps(result.data, indent=2),
            result_message=result.message,
            action_history=formatted_history,
            context_info=context_info,
            conversation_history=formatted_conv
        )

        # Get LLM decision on next action
        response = prompt_model(prompt, model=model, system_prompt=SYSTEM_PROMPT)

        # Check if LLM suggests stopping action chain and providing direct response
        should_continue = True
        if "STOP_ACTION_CHAIN" in response or "PROVIDE_DIRECT_RESPONSE" in response:
            should_continue = False

        # Parse the response to get the next action
        action_type, params = self._parse_action_response(response)

        next_action = Action(
            type=action_type,
            parameters=params
        )

        return next_action, should_continue

    def _generate_user_reply(self, action_history: List, context: ActionContext,
                             messages: List[Dict], model: str) -> str:
        """
        Generate a reply to the user based on the action history and context.

        Args:
            action_history: The history of actions and results
            context: The current context
            messages: The conversation history
            model: The LLM to use

        Returns:
            str: The reply to send to the user
        """
        # Format conversation and action history for the LLM
        formatted_conv_history = self._format_conversation_history(messages)
        formatted_action_history = self._format_action_history(action_history)

        # Format context information
        context_info = self._format_context_for_decision(context)

        # Prepare prompt for user reply
        prompt = USER_REPLY_PROMPT.format(
            conversation_history=formatted_conv_history,
            action_history=formatted_action_history,
            context_info=context_info
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
            formatted += f"PARAMETERS: {json.dumps(action.parameters)}\n"
            formatted += f"RESULT: {'Success' if result.success else 'Failed'}\n"
            formatted += f"DATA: {json.dumps(result.data)}\n"
            formatted += f"MESSAGE: {result.message}\n\n"
        return formatted

    def _parse_action_response(self, response: str) -> Tuple[ActionType, Dict[str, Any]]:
        """
        Parse the LLM response to extract action type and parameters.

        The response should be in the format: ►ACTION_TYPE◄ followed by parameters

        Returns:
            Tuple[ActionType, Dict[str, Any]]: Action type and parameters
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
