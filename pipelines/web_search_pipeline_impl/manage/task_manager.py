"""
Enhanced Task Manager for the Private Web Search Assistant.

This module manages search tasks, chat associations, and updates,
integrating with the Knowledge Graph for improved context awareness.
"""

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, List, Optional, Any

from integration.data.config import DEFAULT_LLM, DEFAULT_PIPELINE
from integration.net.open_webui.api import (
    update_chat_on_server,
    create_new_chat,
    prompt_model,
    CHATS
)
from integration.net.www.chrome.chrome_surfer import search_web
from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants import (
    DEFAULT_MAX_WORKERS, MSG_SEARCH_STARTED, MSG_SEARCH_COMPLETED,
    MSG_SEARCH_CANCELLED, MSG_SEARCH_FAILED, MSG_SEARCH_AUGMENTED,
    UI_DEDICATED_CHAT_TITLE
)
from integration.pipelines.pipelines.web_search_pipeline_impl.manage.kg_manager import (
    KnowledgeGraphManager
)
from integration.pipelines.pipelines.web_search_pipeline_impl.data.prompts import (
    generate_search_response_prompt,
    generate_task_status_prompt,
    generate_results_summary_prompt,
    generate_augment_search_prompt,
    generate_cancel_task_prompt
)
from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants import (
    IntentType, TaskStatus
)


class TaskManager:
    """
    Enhanced Task Manager that handles search tasks, chat associations, and updates.
    Integrates with the Knowledge Graph for improved context awareness.
    """

    def __init__(self, default_model=DEFAULT_LLM, pipeline=DEFAULT_PIPELINE, max_workers=DEFAULT_MAX_WORKERS):
        """
        Initialize the Task Manager.

        Args:
            default_model: Default LLM model ID
            pipeline: Pipeline model ID
            max_workers: Maximum number of worker threads
        """
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize the Knowledge Graph Manager
        self.kg_manager = KnowledgeGraphManager()

        # Chat and task tracking
        self.task_chats = {}  # task_id -> chat_id mapping for dedicated task chats
        self.update_queues = {}  # chat_id -> Queue for updates
        self._update_threads = {}  # chat_id -> Thread for processing updates

        # Model configuration
        self.llm = default_model
        self.pipeline = pipeline

        # Thread and executor management
        self._event_loop = None
        self._event_loop_thread = None
        self._event_loop_started = False
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

    async def initialize(self):
        """Initialize the Task Manager."""
        self.logger.info("Initializing TaskManager")
        self._create_event_loop()

    async def shutdown(self):
        """Shutdown the Task Manager and clean up resources."""
        self.logger.info("Shutting down TaskManager")

        # Stop all update threads
        for chat_id, thread in self._update_threads.items():
            if thread.is_alive():
                self.update_queues[chat_id].put(None)  # Signal to stop
                thread.join(timeout=1.0)

        # Stop the event loop thread if running
        if self._event_loop and self._event_loop_thread and self._event_loop_thread.is_alive():
            self._event_loop.call_soon_threadsafe(self._event_loop.stop)
            self._event_loop_thread.join(timeout=1.0)
            self._event_loop.close()

        # Clean up resources
        self._event_loop_thread = None
        self._event_loop = None
        self._event_loop_started = False
        self.executor.shutdown(wait=False)

    def _create_event_loop(self):
        """Create and start the event loop in a separate thread."""
        if self._event_loop_started:
            return

        # Create a new event loop and run it in a separate thread
        self._event_loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(self._event_loop)
            self._event_loop.run_forever()

        self._event_loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._event_loop_thread.start()
        self._event_loop_started = True

    def add_user_message(self, chat_id: str, content: str) -> str:
        """
        Add a user message to the conversation history.

        Args:
            chat_id: The chat ID
            content: The message content

        Returns:
            The message ID
        """
        message_id = f"msg-{str(uuid.uuid4())}"
        try:
            message_node = self.kg_manager.add_message(chat_id, "user", content)
            return message_node.message_id
        except Exception as e:
            self.logger.error(f"Error adding user message: {str(e)}")
            # Create a message node directly if KG insertion fails
            try:
                # Try to add a minimal context entry if it doesn't exist
                context = self.kg_manager.get_or_create_context(chat_id)
                self.logger.info(f"Created fallback context for chat {chat_id}")
            except Exception as context_err:
                self.logger.error(f"Failed to create fallback context: {str(context_err)}")
            return message_id

    def add_assistant_message(self, chat_id: str, content: str) -> str:
        """
        Add an assistant message to the conversation history.

        Args:
            chat_id: The chat ID
            content: The message content

        Returns:
            The message ID
        """
        try:
            message_node = self.kg_manager.add_message(chat_id, "assistant", content)
            return message_node.message_id
        except Exception as e:
            self.logger.error(f"Error adding assistant message: {str(e)}")
            # Return a generated message ID even if KG insertion fails
            return f"msg-{str(uuid.uuid4())}"

    def create_search_task(self, chat_id: str, search_terms: List[str],
                           semantic_patterns: Optional[List[str]] = None,
                           instructions: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new search task.

        Args:
            chat_id: The chat ID
            search_terms: List of search terms
            semantic_patterns: Optional list of semantic patterns
            instructions: Optional search instructions

        Returns:
            Dictionary with task information
        """
        # Create the task in the knowledge graph
        task = self.kg_manager.create_search_task(
            chat_id=chat_id,
            search_terms=search_terms,
            semantic_patterns=semantic_patterns,
            instructions=instructions
        )

        # Return task information
        return {
            "task_id": task.task_id,
            "search_terms": task.search_terms,
            "semantic_patterns": task.semantic_patterns,
            "instructions": task.instructions,
            "status": task.status
        }

    def create_task_dedicated_chat(self, parent_chat_id: str, task_id: str) -> str:
        """
        Create a dedicated chat for a specific search task.

        Args:
            parent_chat_id: The original chat ID
            task_id: The task ID

        Returns:
            The new chat ID
        """
        # Get the task from the knowledge graph
        task = self.kg_manager.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Create a descriptive title for the new chat
        search_terms_str = ", ".join(task.search_terms)
        title = UI_DEDICATED_CHAT_TITLE.format(search_terms=search_terms_str[:50] + ('...' if len(search_terms_str) > 50 else ''))

        # Create the new chat with the pipeline model ID
        new_chat_data = create_new_chat(model=self.pipeline, title=title)
        new_chat_id = new_chat_data["id"]

        # Add chat data to our local cache
        if new_chat_id not in CHATS:
            CHATS[new_chat_id] = new_chat_data["chat"]

        # Associate the task with the new chat
        self.task_chats[task.task_id] = new_chat_id

        # Update the task with the parent_chat_id
        self.kg_manager.update_search_task(
            task_id=task.task_id,
            properties={"parent_chat_id": parent_chat_id}
        )

        # Create intro message for the dedicated chat
        intro_message = (
            f"# Search Task: {search_terms_str}\n\n"
            f"This is a dedicated chat for tracking your search on: **{search_terms_str}**\n\n"
            f"Task ID: `{task.task_id}`\n"
        )

        if task.semantic_patterns:
            patterns_str = ", ".join(task.semantic_patterns)
            intro_message += f"Patterns: {patterns_str}\n"

        if task.instructions:
            intro_message += f"Instructions: {task.instructions}\n"

        intro_message += "\nI'll post updates about each search term individually as they become available."

        # Send the intro message properly as a system message
        self.send_message_to_chat(new_chat_id, intro_message)

        # Start the update thread for this chat
        if new_chat_id not in self.update_queues:
            self._start_update_thread(new_chat_id)

        return new_chat_id

    def augment_search_task(self, task_id: str, new_terms: List[str],
                            new_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Augment an existing search task with new terms and patterns.

        Args:
            task_id: The task ID
            new_terms: New search terms to add
            new_patterns: New semantic patterns to add

        Returns:
            Dictionary with task information
        """
        # Get the original task
        original_task = self.kg_manager.get_task(task_id)
        if not original_task:
            raise ValueError(f"Task not found: {task_id}")

        # Augment the task
        task = self.kg_manager.augment_search_task(task_id, new_terms, new_patterns)

        # If original task was completed, we've created a new task
        if original_task.status == TaskStatus.COMPLETED:
            # Schedule the new task for execution
            dedicated_chat_id = self.create_task_dedicated_chat(original_task.chat_id, task.task_id)
            self.schedule_task_execution(task.task_id, dedicated_chat_id)

            # Send a notification about the new task
            notification = MSG_SEARCH_AUGMENTED.format(terms=", ".join(new_terms))
            self.send_message_to_chat(original_task.chat_id, notification)
        else:
            # The original task was augmented
            # Send notification to the dedicated chat if it exists
            dedicated_chat_id = self.task_chats.get(task_id)
            if dedicated_chat_id:
                notification = MSG_SEARCH_AUGMENTED.format(terms=", ".join(new_terms))
                self.send_message_to_chat(dedicated_chat_id, notification)

        # Return task information
        return {
            "task_id": task.task_id,
            "search_terms": task.search_terms,
            "semantic_patterns": task.semantic_patterns,
            "instructions": task.instructions,
            "status": task.status,
            "is_new_task": original_task.status == TaskStatus.COMPLETED
        }

    def _start_update_thread(self, chat_id: str):
        """
        Start a thread to process updates for a chat session.

        Args:
            chat_id: The chat ID
        """
        if chat_id in self._update_threads and self._update_threads[chat_id].is_alive():
            return  # Thread already running

        queue = Queue()
        self.update_queues[chat_id] = queue

        def update_worker():
            while True:
                update = queue.get()
                if update is None:  # Signal to stop
                    break

                try:
                    # Send the update
                    self.send_message_to_chat(chat_id, update)
                except Exception as e:
                    self.logger.error(f"Failed to send update: {str(e)}")
                finally:
                    queue.task_done()

        thread = threading.Thread(target=update_worker, daemon=True)
        thread.start()
        self._update_threads[chat_id] = thread

    def send_user_update(self, chat_id: str, message: str):
        """
        Queue an update to be sent to the user.

        Args:
            chat_id: The chat ID
            message: The message to send
        """
        if chat_id not in self.update_queues:
            self._start_update_thread(chat_id)
        self.update_queues[chat_id].put(message)

    def send_message_to_chat(self, chat_id: str, message: str):
        """
        Send a message to a chat without queueing.

        Args:
            chat_id: The chat ID
            message: The message to send
        """
        try:
            # Make sure we have the chat in our cache
            if chat_id not in CHATS:
                # Try to get the chat data
                self.logger.warning(f"Chat {chat_id} not found in cache. Creating a new entry.")
                # Create a minimal chat structure
                CHATS[chat_id] = {
                    "id": chat_id,
                    "models": [self.pipeline],
                    "history": {
                        "messages": {},
                        "currentId": None
                    },
                    "messages": []
                }

            chat_data = CHATS[chat_id]

            # Get the model ID from the chat data if available, otherwise use the pipeline ID
            model_id = self.pipeline
            if "models" in chat_data and len(chat_data["models"]) > 0 and chat_data["models"][0]:
                model_id = chat_data["models"][0]

            # Ensure we're not using the LLM ID for messages in pipeline chats
            if model_id == self.llm:
                model_id = self.pipeline

            # Create a message ID for the new message
            message_id = str(uuid.uuid4())

            # Create a message object
            now_secs = int(time.time())
            msg_obj = {
                "id": message_id,
                "parentId": chat_data["history"]["currentId"],
                "childrenIds": [],
                "role": "assistant",
                "content": message,
                "model": model_id,
                "modelIdx": 0,
                "userContext": None,
                "timestamp": now_secs
            }

            # Add to chat history
            chat_data["history"]["messages"][message_id] = msg_obj
            chat_data["history"]["currentId"] = message_id

            # Add to messages list
            chat_data["messages"].append(msg_obj)

            # Update the chat on the server
            update_chat_on_server(chat_id, chat_data)

            # Add the message to the knowledge graph
            self.kg_manager.add_message(chat_id, "assistant", message)
        except Exception as e:
            self.logger.error(f"Failed to send message to chat {chat_id}: {str(e)}")

    def cancel_task(self, chat_id: str, task_id: str) -> bool:
        """
        Cancel a specific task.

        Args:
            chat_id: The chat ID
            task_id: The task ID

        Returns:
            True if the task was cancelled, False otherwise
        """
        # Cancel the task in the knowledge graph
        success = self.kg_manager.cancel_task(task_id)

        if success:
            # Get the task for notifications
            task = self.kg_manager.get_task(task_id)

            # Notify the original chat
            prompt = generate_cancel_task_prompt(
                task=self.kg_manager.get_task_results_dict(task_id),
                was_successful=True
            )
            response, _ = prompt_model(
                message=prompt,
                chat_id=None,
                model=self.llm,
                transient=True
            )
            self.send_message_to_chat(chat_id, response)

            # If there's a dedicated chat for this task, notify it as well
            dedicated_chat_id = self.task_chats.get(task_id)
            if dedicated_chat_id:
                self.send_message_to_chat(
                    dedicated_chat_id,
                    f"🛑 Task {task_id} has been cancelled by the user from another chat."
                )

        return success

    def cancel_all_tasks(self, chat_id: str) -> int:
        """
        Cancel all tasks associated with a chat.

        Args:
            chat_id: The chat ID

        Returns:
            Number of tasks cancelled
        """
        # Get all active tasks before cancellation
        active_tasks = self.kg_manager.get_active_tasks(chat_id)
        task_ids = [task.task_id for task in active_tasks]

        # Cancel all tasks in the knowledge graph
        count = self.kg_manager.cancel_all_tasks(chat_id)

        if count > 0:
            # Notify the original chat
            self.send_message_to_chat(
                chat_id,
                f"Cancelled {count} active search tasks."
            )

            # Notify all dedicated chats for the cancelled tasks
            for task_id in task_ids:
                dedicated_chat_id = self.task_chats.get(task_id)
                if dedicated_chat_id:
                    self.send_message_to_chat(
                        dedicated_chat_id,
                        f"🛑 Task {task_id} has been cancelled by the user from another chat."
                    )

        return count

    def get_task_status(self, chat_id: str, task_id: str, is_dedicated_chat: bool = False) -> str:
        """
        Get a formatted status message for a task.

        Args:
            chat_id: The chat ID
            task_id: The task ID
            is_dedicated_chat: Whether this is for the dedicated task chat

        Returns:
            Formatted status message
        """
        # Get the task details
        task_dict = self.kg_manager.get_task_results_dict(task_id)
        if not task_dict:
            return f"No task found with ID: {task_id}"

        # Check if this is the dedicated chat for this task
        if not is_dedicated_chat:
            is_dedicated_chat = self.is_dedicated_chat(chat_id, task_id)

        # Generate a status response using the LLM
        prompt = generate_task_status_prompt(
            task=task_dict,
            is_dedicated_chat=is_dedicated_chat
        )

        response, _ = prompt_model(
            message=prompt,
            chat_id=None,
            model=self.llm,
            transient=True
        )

        return response

    def get_tasks_status_summary(self, chat_id: str) -> str:
        """
        Get a summary of all tasks for a chat.

        Args:
            chat_id: The chat ID

        Returns:
            Formatted status summary
        """
        # Get tasks from the knowledge graph
        active_tasks = self.kg_manager.get_active_tasks(chat_id)
        completed_tasks = self.kg_manager.get_completed_tasks(chat_id)

        if not active_tasks and not completed_tasks:
            return "There are no search tasks associated with this chat."

        # Format the status report
        status = "# Search Tasks Status\n\n"

        if active_tasks:
            status += "## Active Tasks\n\n"
            for task in active_tasks:
                status += f"- **{', '.join(task.search_terms)}** ({task.status})\n"
                status += f"  - Task ID: `{task.task_id}`\n"
                status += f"  - Duration: {task.duration():.1f}s\n\n"

        if completed_tasks:
            status += "## Completed Tasks\n\n"
            for task in completed_tasks:
                status += f"- **{', '.join(task.search_terms)}**\n"
                status += f"  - Task ID: `{task.task_id}`\n"
                status += f"  - Results: {len(task.results)}, Patterns: {len(task.discovered_patterns)}\n"
                status += f"  - Duration: {task.duration():.1f}s\n\n"

        status += "Use `show results for [task ID]` to see detailed results for a specific task."

        return status

    def get_results_summary(self, chat_id: str, task_id: str) -> str:
        """
        Get a summary of the results for a task.

        Args:
            chat_id: The chat ID
            task_id: The task ID

        Returns:
            Formatted results summary
        """
        # Get the task details
        task_dict = self.kg_manager.get_task_results_dict(task_id)
        if not task_dict:
            return f"No task found with ID: {task_id}"

        # Check if the task is completed
        if task_dict["status"] != TaskStatus.COMPLETED:
            return f"Task {task_id} is not completed yet (Status: {task_dict['status']}). Results are not available."

        # Get related tasks if any
        related_tasks = []
        for related_id in task_dict.get("related_tasks", []):
            related_dict = self.kg_manager.get_task_results_dict(related_id)
            if related_dict:
                related_tasks.append(related_dict)

        # Generate a results summary using the LLM
        prompt = generate_results_summary_prompt(
            task=task_dict,
            include_results=True,
            related_tasks=related_tasks
        )

        response, _ = prompt_model(
            message=prompt,
            chat_id=None,
            model=self.llm,
            transient=True
        )

        return response

    def is_dedicated_chat(self, chat_id: str, task_id: str) -> bool:
        """
        Check if a chat is the dedicated chat for a task.

        Args:
            chat_id: The chat ID
            task_id: The task ID

        Returns:
            True if the chat is the dedicated chat for the task, False otherwise
        """
        return self.task_chats.get(task_id) == chat_id

    def schedule_task_execution(self, task_id: str, dedicated_chat_id: str):
        """
        Schedule the execution of a search task.

        Args:
            task_id: The task ID
            dedicated_chat_id: The dedicated chat ID for detailed updates
        """
        if not self._event_loop_started:
            self._create_event_loop()

        self._event_loop.call_soon_threadsafe(
            asyncio.create_task,
            self._run_task_with_updates(task_id, dedicated_chat_id)
        )

    async def _run_task_with_updates(self, task_id: str, dedicated_chat_id: str):
        """
        Run a search task and send updates to the dedicated chat.

        Args:
            task_id: The task ID
            dedicated_chat_id: The dedicated chat ID
        """
        # Get the task from the knowledge graph with proper error handling
        task = self.kg_manager.get_task(task_id)
        if not task:
            self.logger.error(f"Task not found: {task_id}")
            self.send_message_to_chat(dedicated_chat_id, f"❌ Error: Task {task_id} could not be found or accessed.")
            return

        # Validate that task has all required attributes
        required_attrs = ['task_id', 'search_terms', 'status']
        for attr in required_attrs:
            if not hasattr(task, attr):
                self.logger.error(f"Task {task_id} missing required attribute: {attr}")
                self.send_message_to_chat(dedicated_chat_id, f"❌ Error: Task {task_id} has invalid structure. Missing {attr}.")
                return

        try:
            # Mark the task as running with error handling
            try:
                self.kg_manager.update_search_task(
                    task_id=task_id,
                    properties={"status": TaskStatus.RUNNING, "start_time": time.time()}
                )
            except Exception as update_error:
                self.logger.error(f"Error updating task to running state: {str(update_error)}")
                # Continue anyway, as this is not critical

            # Send initial update
            initial_message = MSG_SEARCH_STARTED.format(terms=", ".join(task.search_terms))
            self.send_message_to_chat(dedicated_chat_id, initial_message)

            # Run the search
            results = []
            patterns = []

            # Get the updated task after marking it as running
            task = self.kg_manager.get_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found after updating to running state")

            # Execute searches in a thread pool - one search per term
            loop = asyncio.get_running_loop()

            for i, term in enumerate(task.search_terms, 1):
                # Check if the task was cancelled
                task = self.kg_manager.get_task(task_id)
                if task and task.status == TaskStatus.CANCELLED:
                    break

                # Update on progress
                term_progress_msg = f"### 🔍 Processing search term {i}/{len(task.search_terms)}: '{term}'"
                self.send_message_to_chat(dedicated_chat_id, term_progress_msg)

                # Execute the individual search with error handling for each term
                try:
                    term_results, term_patterns = await loop.run_in_executor(
                        self.executor,
                        lambda: search_web(
                            search_terms=[term],  # Just the current term
                            semantic_patterns=task.semantic_patterns if hasattr(task, 'semantic_patterns') else None,
                            instructions=task.instructions if hasattr(task, 'instructions') else None,
                            max_workers=self.max_workers
                        )
                    )

                    # Add to the aggregate results
                    results.extend(term_results)
                    patterns.extend(term_patterns)

                    # Send per-term update to dedicated chat
                    term_summary = f"### ✅ Search for '{term}' completed\n"
                    term_summary += f"Found {len(term_results)} results and {len(term_patterns)} patterns"
                    self.send_message_to_chat(dedicated_chat_id, term_summary)

                    # If we have results, show a sample
                    if term_results:
                        sample_results = f"Sample results for '{term}':\n"
                        for r in term_results[:3]:  # Show only 3 sample results
                            sample_results += f"- {r}\n"
                        if len(term_results) > 3:
                            sample_results += f"...and {len(term_results) - 3} more."
                        self.send_message_to_chat(dedicated_chat_id, sample_results)

                except Exception as e:
                    error_msg = f"### ❌ Error searching for term '{term}': {str(e)}"
                    self.send_message_to_chat(dedicated_chat_id, error_msg)
                    # Continue with other terms despite the error

            # Check if the task was cancelled during execution
            try:
                task = self.kg_manager.get_task(task_id)
                if task and task.status == TaskStatus.CANCELLED:
                    cancel_msg = MSG_SEARCH_CANCELLED.format(task_id=task_id)
                    self.send_message_to_chat(dedicated_chat_id, cancel_msg)
                    return
            except Exception as check_error:
                self.logger.error(f"Error checking if task was cancelled: {str(check_error)}")
                # Continue anyway to try to save results

            # Mark the task as completed with results
            try:
                self.kg_manager.complete_task(task_id, results, patterns)
            except Exception as complete_error:
                self.logger.error(f"Error completing task: {str(complete_error)}")
                # Try a fallback method to save at least some results
                try:
                    self.kg_manager.update_search_task(
                        task_id=task_id,
                        properties={
                            "status": TaskStatus.COMPLETED,
                            "end_time": time.time(),
                            "results": results,
                            "discovered_patterns": patterns
                        }
                    )
                except Exception as fallback_error:
                    self.logger.error(f"Fallback completion also failed: {str(fallback_error)}")

            # Send completion update
            try:
                task = self.kg_manager.get_task(task_id)
                summary = MSG_SEARCH_COMPLETED.format(
                    duration=task.duration() if hasattr(task, 'duration') and callable(task.duration) else 0.0,
                    result_count=len(results),
                    pattern_count=len(patterns)
                )
                self.send_message_to_chat(dedicated_chat_id, summary)
            except Exception as summary_error:
                self.logger.error(f"Error generating completion summary: {str(summary_error)}")
                # Send a simple summary instead
                self.send_message_to_chat(dedicated_chat_id,
                                          f"✅ Search completed. Found {len(results)} results and {len(patterns)} patterns.")

            # Generate a detailed summary
            detailed_summary = (
                "# Search Complete\n\n"
                f"### ✅ All search terms processed\n\n"
                f"Total results: {len(results)}\n"
                f"Total patterns discovered: {len(patterns)}\n\n"
                "A detailed summary will be generated shortly."
            )
            self.send_message_to_chat(dedicated_chat_id, detailed_summary)

            # Allow a short delay for processing
            await asyncio.sleep(2)

            # Generate a final results summary using the LLM
            try:
                task_dict = self.kg_manager.get_task_results_dict(task_id)
                prompt = generate_results_summary_prompt(
                    task=task_dict,
                    include_results=True
                )

                summary, _ = prompt_model(
                    message=prompt,
                    chat_id=None,
                    model=self.llm,
                    transient=True
                )

                # Send the final summary
                self.send_message_to_chat(dedicated_chat_id, summary)
            except Exception as llm_error:
                self.logger.error(f"Error generating LLM summary: {str(llm_error)}")
                # Send a simple results message instead
                simple_results = "# Search Results\n\n"
                for i, result in enumerate(results[:10], 1):
                    simple_results += f"{i}. {result}\n"
                if len(results) > 10:
                    simple_results += f"\n... and {len(results) - 10} more results."
                self.send_message_to_chat(dedicated_chat_id, simple_results)

        except asyncio.CancelledError:
            # Mark the task as cancelled
            self.kg_manager.cancel_task(task_id)
            cancel_msg = MSG_SEARCH_CANCELLED.format(task_id=task_id)
            self.send_message_to_chat(dedicated_chat_id, cancel_msg)

        except Exception as e:
            # Mark the task as failed
            error_msg = str(e)
            try:
                self.kg_manager.fail_task(task_id, error_msg)
            except Exception as fail_error:
                self.logger.error(f"Error marking task as failed: {str(fail_error)}")

            fail_msg = MSG_SEARCH_FAILED.format(error=error_msg)
            self.send_message_to_chat(dedicated_chat_id, fail_msg)

    def process_intent(self, chat_id: str, message_id: str, intent_type: IntentType,
                       parameters: Dict[str, Any], confidence: float = 1.0) -> Dict[str, Any]:
        """
        Process a detected intent from a user message.

        Args:
            chat_id: The chat ID
            message_id: The message ID that generated this intent
            intent_type: The type of intent detected
            parameters: Parameters for the intent
            confidence: Confidence score for the intent

        Returns:
            Dictionary with response information
        """
        # Add the intent to the knowledge graph
        self.kg_manager.add_intent(
            message_id=message_id,
            chat_id=chat_id,
            intent_type=intent_type,
            confidence=confidence,
            parameters=parameters
        )

        # Process the intent based on its type
        if intent_type == IntentType.SEARCH:
            return self._process_search_intent(chat_id, parameters)
        elif intent_type == IntentType.FOLLOW_UP:
            return self._process_follow_up_intent(chat_id, parameters)
        elif intent_type == IntentType.AUGMENT_SEARCH:
            return self._process_augment_search_intent(chat_id, parameters)
        elif intent_type == IntentType.STATUS:
            return self._process_status_intent(chat_id, parameters)
        elif intent_type == IntentType.CANCEL:
            return self._process_cancel_intent(chat_id, parameters)
        elif intent_type == IntentType.HELP:
            return self._process_help_intent(chat_id, parameters)
        elif intent_type == IntentType.CHAT:
            return self._process_chat_intent(chat_id, parameters)
        else:
            return {"response": "I'm not sure how to help with that. Type 'help' for assistance."}

    def _process_search_intent(self, chat_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a search intent.

        Args:
            chat_id: The chat ID
            parameters: Intent parameters

        Returns:
            Dictionary with response information
        """
        search_terms = parameters.get("search_terms", [])
        semantic_patterns = parameters.get("semantic_patterns")
        instructions = parameters.get("instructions")

        # Create the search task
        task = self.create_search_task(
            chat_id=chat_id,
            search_terms=search_terms,
            semantic_patterns=semantic_patterns,
            instructions=instructions
        )

        # Create a dedicated chat for the task
        dedicated_chat_id = self.create_task_dedicated_chat(
            parent_chat_id=chat_id,
            task_id=task["task_id"]
        )

        # Schedule the task for execution
        self.schedule_task_execution(task["task_id"], dedicated_chat_id)

        # Generate a response using the LLM
        prompt = generate_search_response_prompt(
            search_terms=search_terms,
            task_id=task["task_id"],
            is_dedicated_chat=False,
            patterns=semantic_patterns,
            instructions=instructions
        )

        response, _ = prompt_model(
            message=prompt,
            chat_id=None,
            model=self.llm,
            transient=True
        )

        return {"response": response, "task_id": task["task_id"]}

    def _process_follow_up_intent(self, chat_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a follow-up intent.

        Args:
            chat_id: The chat ID
            parameters: Intent parameters

        Returns:
            Dictionary with response information
        """
        # This is a special case where we might have a direct answer or need to create a search
        if "direct_response" in parameters:
            return {"response": parameters["direct_response"]}

        # Otherwise, treat it as a new search with the inferred terms
        search_terms = parameters.get("search_terms", [])
        return self._process_search_intent(chat_id, {"search_terms": search_terms})

    def _process_augment_search_intent(self, chat_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an augment search intent.

        Args:
            chat_id: The chat ID
            parameters: Intent parameters

        Returns:
            Dictionary with response information
        """
        task_id = parameters.get("task_id")
        new_terms = parameters.get("search_terms", [])
        new_patterns = parameters.get("semantic_patterns")

        if not task_id:
            # We need a task ID to augment
            recent_tasks = self.kg_manager.get_tasks_for_chat(chat_id, limit=1)
            if not recent_tasks:
                return {"response": "I'm not sure which search you want to augment. Please specify a task ID."}
            task_id = recent_tasks[0].task_id

        # Get the original task
        task = self.kg_manager.get_task(task_id)
        if not task:
            return {"response": f"I couldn't find a task with ID {task_id}. Please check the ID and try again."}

        # Augment the search task
        result = self.augment_search_task(
            task_id=task_id,
            new_terms=new_terms,
            new_patterns=new_patterns
        )

        # Generate a response using the LLM
        prompt = generate_augment_search_prompt(
            task_id=task_id,
            original_terms=task.search_terms,
            new_terms=new_terms,
            original_patterns=task.semantic_patterns,
            new_patterns=new_patterns,
            task_status=task.status
        )

        response, _ = prompt_model(
            message=prompt,
            chat_id=None,
            model=self.llm,
            transient=True
        )

        return {"response": response, "task_id": result.get("task_id")}

    def _process_status_intent(self, chat_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a status intent.

        Args:
            chat_id: The chat ID
            parameters: Intent parameters

        Returns:
            Dictionary with response information
        """
        task_id = parameters.get("task_id")

        if task_id:
            # Get status for a specific task
            response = self.get_task_status(chat_id, task_id)
        else:
            # Get status for all tasks
            response = self.get_tasks_status_summary(chat_id)

        return {"response": response}

    def _process_cancel_intent(self, chat_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a cancel intent.

        Args:
            chat_id: The chat ID
            parameters: Intent parameters

        Returns:
            Dictionary with response information
        """
        task_id = parameters.get("task_id")

        if task_id:
            # Cancel a specific task
            success = self.cancel_task(chat_id, task_id)

            if success:
                return {"response": f"Task {task_id} has been cancelled successfully."}
            else:
                return {"response": f"Could not cancel task {task_id}. It may not exist, not be associated with this chat, or is already completed."}
        else:
            # Cancel all tasks
            count = self.cancel_all_tasks(chat_id)
            return {"response": f"Cancelled {count} active search tasks."}

    def _process_help_intent(self, chat_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a help intent.

        Args:
            chat_id: The chat ID
            parameters: Intent parameters

        Returns:
            Dictionary with response information
        """
        # If a direct response was provided, use it
        if "direct_response" in parameters:
            return {"response": parameters["direct_response"]}

        # Otherwise, generate a generic help message
        return {"response": """
# Private Web Search Assistant - Help

I can help you search the web privately and efficiently. Here's how to use me:

## Basic Searching
Simply type your search query, for example:
- "Search for climate change effects on agriculture"
- "Find information about quantum computing"

## Follow-up Questions
After a search, you can ask follow-up questions like:
- "What about in tropical regions?"
- "How does it impact developing countries?"

## Augmenting Searches
Add to an existing search with:
- "Also check for drought impacts"
- "Add weather patterns to that search"

## Managing Searches
- Check status: "What's the status of my search?" or "status of task [ID]"
- View results: "Show me the results" or "results for task [ID]"
- Cancel: "Cancel my search" or "cancel task [ID]"

Each search creates a dedicated chat where you'll get detailed updates and results.
"""}

    def _process_chat_intent(self, chat_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a chat intent (simple questions, greetings, etc.).

        Args:
            chat_id: The chat ID
            parameters: Intent parameters

        Returns:
            Dictionary with response information
        """
        # Just return the direct response from the parameters
        return {"response": parameters.get("direct_response", "I'm here to help you search the web. What would you like to find?")}
