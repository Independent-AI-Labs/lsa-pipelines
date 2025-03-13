"""
Private Web Search Pipeline for intelligent analysis of user queries and task management.
"""

from typing import List, Union, Generator, Iterator, Dict, Any, Optional

from integration.data.config import DEFAULT_LLM
from integration.net.open_webui.api import prompt_model
from integration.pipelines.pipelines.web_search_pipeline_impl.prompts import (
    generate_intent_analysis_prompt,
    generate_search_response_prompt,
    generate_context_answer_prompt,
    generate_task_status_prompt,
    generate_help_prompt,
    generate_results_summary_prompt
)
from integration.pipelines.pipelines.web_search_pipeline_impl.task_manager import TaskManager


class Pipeline:
    """
    Private Web Search Pipeline that intelligently analyzes user queries,
    manages search tasks, and provides appropriate responses.
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.name = "Private Web Search Assistant"
        self.task_manager = TaskManager()
        self.default_model = DEFAULT_LLM

    async def on_startup(self):
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")
        await self.task_manager.initialize()

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")
        await self.task_manager.shutdown()

    def _extract_chat_info(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant information from the request body.

        Args:
            body: The request body

        Returns:
            Dictionary with chat information
        """
        chat_id = body.get("chat_id", "unknown")
        user_info = body.get("user", {})
        user_id = user_info.get("id", "unknown")

        return {
            "chat_id": chat_id,
            "user_id": user_id,
            "username": user_info.get("name", "User")
        }

    def _analyze_user_intent_with_llm(self, user_message: str, chat_id: str, model: str) -> Dict[str, Any]:
        """
        Use the LLM to analyze the user's message and determine the appropriate action.

        Args:
            user_message: The user's message
            chat_id: The chat ID where the message was sent
            model: The model ID to use for LLM prompting

        Returns:
            A dictionary with the determined action and parameters
        """
        # Get context information
        recent_messages = self.task_manager.get_recent_messages(chat_id)

        # Get tasks associated with this chat
        tasks = self.task_manager.get_tasks_for_chat(chat_id)

        # Generate the prompt
        prompt = generate_intent_analysis_prompt(
            user_message=user_message,
            recent_messages=recent_messages,
            active_tasks=tasks,
            chat_tasks=self.task_manager.chat_tasks.get(chat_id, [])
        )

        # Get the LLM's analysis
        llm_response, _ = prompt_model(
            message=prompt,
            chat_id=None,  # Create a transient chat
            model=model,
            transient=True  # Don't persist the chat
        )

        # Parse the LLM's response
        return self._parse_intent_analysis(llm_response, chat_id, user_message)

    def _parse_intent_analysis(self, llm_response: str, chat_id: str, user_message: str) -> Dict[str, Any]:
        """
        Parse the LLM's intent analysis response and extract the action and parameters.

        Args:
            llm_response: The LLM's response
            chat_id: The chat ID
            user_message: The original user message

        Returns:
            A dictionary with the action and parameters
        """
        lines = llm_response.strip().split("\n")

        # Initialize result with default values
        result = {
            "action": "help",  # Default action
            "chat_id": chat_id
        }

        # Extract information from the LLM's response
        for line in lines:
            line = line.strip()

            if line.startswith("ACTION:"):
                action = line[len("ACTION:"):].strip().lower()
                result["action"] = action

            elif line.startswith("TASK_ID:"):
                task_id = line[len("TASK_ID:"):].strip()
                if task_id.lower() != "none":
                    result["task_id"] = task_id

            elif line.startswith("SEARCH_TERMS:"):
                terms_str = line[len("SEARCH_TERMS:"):].strip()
                if terms_str.lower() != "none":
                    result["search_terms"] = [term.strip() for term in terms_str.split(",")]
                else:
                    result["search_terms"] = [user_message]

            elif line.startswith("PATTERNS:"):
                patterns_str = line[len("PATTERNS:"):].strip()
                if patterns_str.lower() != "none":
                    result["semantic_patterns"] = [pattern.strip() for pattern in patterns_str.split(",")]

            elif line.startswith("INSTRUCTIONS:"):
                instructions = line[len("INSTRUCTIONS:"):].strip()
                if instructions.lower() != "none":
                    result["instructions"] = instructions

        # Set default search terms if not provided
        if result["action"] == "create_search_task" and "search_terms" not in result:
            result["search_terms"] = [user_message]

        return result

    def _try_answer_from_context_with_llm(self, chat_id: str, query: str, model: str) -> Optional[str]:
        """
        Use the LLM to attempt to answer the user's query using existing conversation context.

        Args:
            chat_id: The chat ID
            query: The user's query
            model: The model ID to use for LLM prompting

        Returns:
            An answer if possible, None otherwise
        """
        # Get recent messages
        recent_messages = self.task_manager.get_recent_messages(chat_id)
        if not recent_messages:
            return None

        # Generate the prompt
        prompt = generate_context_answer_prompt(
            user_query=query,
            recent_messages=recent_messages,
            chat_tasks=self.task_manager.chat_tasks.get(chat_id, [])
        )

        # Get the LLM's response
        llm_response, _ = prompt_model(
            message=prompt,
            chat_id=None,
            model=model,
            transient=True
        )

        # Parse the response
        lines = llm_response.strip().split("\n")
        can_answer = False
        answer = None

        for line in lines:
            line = line.strip()

            if line.startswith("CAN_ANSWER:"):
                can_answer_str = line[len("CAN_ANSWER:"):].strip().lower()
                can_answer = can_answer_str == "yes"

            elif line.startswith("ANSWER:"):
                answer_text = line[len("ANSWER:"):].strip()
                if answer_text:
                    answer = answer_text

        # If there's a multi-line answer (not captured by the simple parsing above)
        if not answer and can_answer:
            answer_section = llm_response.split("ANSWER:", 1)
            if len(answer_section) > 1:
                answer = answer_section[1].strip()

        return answer if can_answer and answer else None

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
        print(f"pipe:{__name__}")
        if user_message.startswith("### Task:"):
            return ""

        # Check if this is just a title request
        if body.get("title", False):
            return self.name

        # Extract chat info
        chat_info = self._extract_chat_info(body)
        chat_id = chat_info["chat_id"]

        yield "<think>"
        yield " "

        # Store the pipeline model_id for dedicated chats and use default_model for LLM operations
        pipeline_model_id = model_id  # This is the model that represents the pipeline itself
        llm = self.default_model  # This is what we'll use for actual LLM operations

        self.task_manager.llm = llm  # Set the LLM model in the task manager

        # Cache the user message
        if chat_id is not None:
            self.task_manager.cache_message(chat_id, "user", user_message)

        try:
            # Analyze the user's intent using the LLM
            intent = self._analyze_user_intent_with_llm(user_message, chat_id, llm)

            # Handle different types of intents
            if intent["action"] == "create_search_task":
                # Create a new search task
                task = self.task_manager.create_task(
                    chat_id=intent["chat_id"],
                    search_terms=intent["search_terms"],
                    semantic_patterns=intent.get("semantic_patterns"),
                    instructions=intent.get("instructions"),
                    user_id=chat_info["user_id"]
                )

                # Create a dedicated chat for this task using the pipeline's model_id
                dedicated_chat_id = self.task_manager.create_task_dedicated_chat(
                    chat_id,
                    task,
                    chat_info["username"],
                    pipeline_model_id=pipeline_model_id  # Use the pipeline's model_id
                )

                # Schedule the task execution
                self.task_manager.schedule_task_execution(task, dedicated_chat_id)

                # Generate a response using the LLM
                prompt = generate_search_response_prompt(
                    search_terms=task.search_terms,
                    patterns=task.semantic_patterns,
                    instructions=task.instructions,
                    task_id=task.task_id
                )

                response, _ = prompt_model(
                    message=prompt,
                    chat_id=None,
                    model=llm,
                    transient=True
                )

                yield "</think>"
                yield ""
                yield response

            elif intent["action"] == "answer_from_context":
                # Try to answer the query from existing context using the LLM
                answer = self._try_answer_from_context_with_llm(chat_id, user_message, llm)
                if answer:
                    yield "</think>"
                    yield answer

                # If we couldn't answer from context, suggest a search
                yield "</think>"
                yield (
                    "I don't have enough information to answer that question based on our conversation. "
                    "Would you like me to perform a web search to find more information? "
                    "If so, please provide more details about what you're looking for."
                )

            elif intent["action"] == "query_existing_task":
                # Handle a query about an existing task
                if "task_id" not in intent:
                    yield "</think>"
                    yield "I need a task ID to answer your query. Please specify which task you're asking about."

                task = self.task_manager.get_task(chat_id, intent["task_id"])
                if not task:
                    yield "</think>"
                    yield f"I couldn't find a task with ID {intent['task_id']}. Please check the ID and try again."

                # Use the LLM to generate a status response
                task_dict = {
                    "task_id": task.task_id,
                    "search_terms": task.search_terms,
                    "status": task.status,
                    "duration": task.duration(),
                    "results": task.results if hasattr(task, 'results') else [],
                    "discovered_patterns": task.discovered_patterns if hasattr(task, 'discovered_patterns') else [],
                    "errors": task.errors if hasattr(task, 'errors') else []
                }

                dedicated_chat_id = self.task_manager.task_chats.get(task.task_id)
                is_dedicated_chat = (chat_id == dedicated_chat_id)

                prompt = generate_task_status_prompt(
                    task=task_dict,
                    is_dedicated_chat=is_dedicated_chat
                )

                response, _ = prompt_model(
                    message=prompt,
                    chat_id=None,
                    model=llm,
                    transient=True
                )

                yield "</think>"
                yield response

            elif intent["action"] == "cancel_task":
                if "task_id" not in intent:
                    yield "</think>"
                    yield "Please specify a task ID to cancel."
                yield "</think>"
                success = self.task_manager.cancel_task(chat_id, intent["task_id"])
                if success:
                    yield "</think>"
                    yield f"Task {intent['task_id']} has been cancelled successfully."
                else:
                    yield "</think>"
                    yield f"Could not cancel task {intent['task_id']}. It may not exist, not be associated with this chat, or is already completed."

            elif intent["action"] == "cancel_all_tasks":
                count = self.task_manager.cancel_all_tasks(chat_id)
                yield "</think>"
                yield f"Cancelled {count} active search tasks."

            elif intent["action"] == "show_results":
                if "task_id" in intent:
                    # Show results for a specific task
                    task = self.task_manager.get_task(chat_id, intent["task_id"])
                    if not task:
                        yield "</think>"
                        yield f"No task found with ID: {intent['task_id']}"

                    if task.status != "completed":
                        yield "</think>"
                        yield f"Task {intent['task_id']} is not completed yet (Status: {task.status}). Results are not available."

                    # Use the LLM to generate a results summary
                    task_dict = {
                        "task_id": task.task_id,
                        "search_terms": task.search_terms,
                        "status": task.status,
                        "duration": task.duration(),
                        "results": task.results if hasattr(task, 'results') else [],
                        "discovered_patterns": task.discovered_patterns if hasattr(task, 'discovered_patterns') else []
                    }

                    prompt = generate_results_summary_prompt(
                        task=task_dict,
                        include_results=True
                    )

                    response, _ = prompt_model(
                        message=prompt,
                        chat_id=None,
                        model=llm,
                        transient=True
                    )

                    yield "</think>"
                    yield response
                else:
                    # Show results for all completed tasks
                    completed_tasks = []

                    for task_id in self.task_manager.chat_tasks.get(chat_id, []):
                        task = self.task_manager.get_task(chat_id, task_id)
                        if task and task.status == "completed":
                            completed_tasks.append({
                                "task_id": task.task_id,
                                "search_terms": task.search_terms,
                                "status": task.status,
                                "duration": task.duration(),
                                "results_count": len(task.results) if hasattr(task, 'results') else 0,
                                "patterns_count": len(task.discovered_patterns) if hasattr(task, 'discovered_patterns') else 0
                            })

                    if not completed_tasks:
                        yield "</think>"
                        yield "There are no completed search tasks associated with this chat."

                    # Format a simple summary
                    summary = "# Completed Search Tasks\n\n"
                    for i, task in enumerate(completed_tasks, 1):
                        summary += f"{i}. **{', '.join(task['search_terms'])}**\n"
                        summary += f"   - Task ID: `{task['task_id']}`\n"
                        summary += f"   - Results: {task['results_count']}, Patterns: {task['patterns_count']}\n"
                        summary += f"   - Duration: {task['duration']:.1f}s\n\n"

                    summary += "Use `show results for [task ID]` to see detailed results for a specific task."

                    yield "</think>"
                    yield summary

            elif intent["action"] == "show_status":
                if "task_id" in intent:
                    # Show status for a specific task
                    task = self.task_manager.get_task(chat_id, intent["task_id"])
                    if not task:
                        yield "</think>"
                        yield f"No task found with ID: {intent['task_id']}"

                    # Use the LLM to generate a status response
                    task_dict = {
                        "task_id": task.task_id,
                        "search_terms": task.search_terms,
                        "status": task.status,
                        "duration": task.duration(),
                        "results": task.results if hasattr(task, 'results') else [],
                        "discovered_patterns": task.discovered_patterns if hasattr(task, 'discovered_patterns') else []
                    }

                    prompt = generate_task_status_prompt(
                        task=task_dict,
                        is_dedicated_chat=False
                    )

                    response, _ = prompt_model(
                        message=prompt,
                        chat_id=None,
                        model=llm,
                        transient=True
                    )

                    yield "</think>"
                    yield response
                else:
                    # Show status for all tasks associated with this chat
                    tasks = self.task_manager.get_tasks_for_chat(chat_id)

                    if not tasks:
                        yield "</think>"
                        yield "There are no search tasks associated with this chat."

                    # Group tasks by status
                    pending = [t for t in tasks if t["status"] == "pending"]
                    running = [t for t in tasks if t["status"] == "running"]
                    completed = [t for t in tasks if t["status"] == "completed"]
                    failed = [t for t in tasks if t["status"] == "failed"]
                    cancelled = [t for t in tasks if t["status"] == "cancelled"]

                    # Format the status report
                    status = "# Search Tasks Status\n\n"

                    if pending or running:
                        status += "## Active Tasks\n\n"
                        for task in pending + running:
                            status += f"- **{', '.join(task['search_terms'])}** ({task['status']})\n"
                            status += f"  - Task ID: `{task['task_id']}`\n"
                            status += f"  - Duration: {task['duration']:.1f}s\n\n"

                    if completed:
                        status += "## Completed Tasks\n\n"
                        for task in completed:
                            status += f"- **{', '.join(task['search_terms'])}**\n"
                            status += f"  - Task ID: `{task['task_id']}`\n"
                            status += f"  - Results: {task['results_count']}, Patterns: {task['patterns_count']}\n"
                            status += f"  - Duration: {task['duration']:.1f}s\n\n"

                    if failed or cancelled:
                        status += "## Failed/Cancelled Tasks\n\n"
                        for task in failed + cancelled:
                            status += f"- **{', '.join(task['search_terms'])}** ({task['status']})\n"
                            status += f"  - Task ID: `{task['task_id']}`\n"
                            status += f"  - Duration: {task['duration']:.1f}s\n\n"

                    yield "</think>"
                    yield status

            elif intent["action"] == "help":
                # Generate a help message using the LLM
                prompt = generate_help_prompt()
                response, _ = prompt_model(
                    message=prompt,
                    chat_id=None,
                    model=llm,
                    transient=True
                )

                yield "</think>"
                yield response

            else:

                yield "</think>"
                yield "I'm not sure how to help with that. You can ask me to search for information, show results, or cancel tasks. Type 'help' for more information."
        except Exception as e:
            print(f"Error processing request: {str(e)}")

            yield "</think>"
            yield f"I encountered an error while processing your request: {str(e)}"

