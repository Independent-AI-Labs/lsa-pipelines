"""
Task Manager for handling search tasks, chat associations, and updates.
"""

import asyncio
import threading
import time
import uuid
from queue import Queue
from typing import List, Dict, Any, Optional

from integration.data.config import DEFAULT_LLM, DEFAULT_PIPELINE
from integration.net.open_webui.api import (
    update_chat_on_server,
    create_new_chat,
    prompt_model,
    CHATS  # Import the CHATS dictionary
)
from integration.pipelines.pipelines.web_search_pipeline_impl.prompts import (
    generate_results_summary_prompt
)
from integration.pipelines.pipelines.web_search_pipeline_impl.utils import SearchManager, SearchTask, TaskStatus


class TaskManager:
    """
    Manages search tasks, chat associations, and updates.
    Handles the lifecycle of search tasks and their dedicated chats.
    """

    def __init__(self, default_model=DEFAULT_LLM, pipeline=DEFAULT_PIPELINE):
        """
        Initialize the task manager.

        Args:
            default_model: Default model ID to use for LLM prompting
        """
        self.search_manager = SearchManager()
        self.update_queues = {}  # chat_id -> Queue for updates
        self._update_threads = {}  # chat_id -> Thread for processing updates
        self._event_loop = None
        self._event_loop_thread = None
        self._event_loop_started = False
        self.task_chats = {}  # task_id -> chat_id mapping for dedicated task chats
        self.chat_tasks = {}  # chat_id -> list of task_ids associated with this chat
        self.conversation_cache = {}  # chat_id -> list of recent messages
        self.llm = default_model
        self.pipeline = pipeline

    async def initialize(self):
        """Initialize the task manager."""
        print("Initializing TaskManager")
        await self.search_manager.initialize()
        self._create_event_loop()

    async def shutdown(self):
        """Shutdown the task manager and clean up resources."""
        print("Shutting down TaskManager")
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

        self._event_loop_thread = None
        self._event_loop = None
        self._event_loop_started = False
        await self.search_manager.shutdown()

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

    def cache_message(self, chat_id: str, role: str, content: str):
        """
        Cache a message for context preservation.

        Args:
            chat_id: The chat ID
            role: The role of the message sender (user/assistant)
            content: The message content
        """
        if chat_id not in self.conversation_cache:
            self.conversation_cache[chat_id] = []

        # Add message to cache
        self.conversation_cache[chat_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

        # Keep only the last 10 messages
        if len(self.conversation_cache[chat_id]) > 10:
            self.conversation_cache[chat_id] = self.conversation_cache[chat_id][-10:]

    def get_recent_messages(self, chat_id: str, count: int = 5) -> List[Dict]:
        """
        Get the most recent messages from the conversation cache.

        Args:
            chat_id: The chat ID
            count: Number of recent messages to retrieve

        Returns:
            List of recent messages
        """
        if chat_id not in self.conversation_cache:
            return []

        return self.conversation_cache[chat_id][-count:]

    def create_task(self, chat_id: str, search_terms: List[str],
                    semantic_patterns: Optional[List[str]] = None,
                    instructions: Optional[str] = None,
                    user_id: str = "unknown") -> SearchTask:
        """
        Create a new search task.

        Args:
            chat_id: The chat ID where the task was created
            search_terms: List of search terms
            semantic_patterns: Optional list of semantic patterns
            instructions: Optional search instructions
            user_id: User ID

        Returns:
            The created search task
        """
        task = self.search_manager.create_task(
            chat_id=chat_id,
            search_terms=search_terms,
            semantic_patterns=semantic_patterns,
            instructions=instructions,
            user_id=user_id
        )

        # Associate the task with the chat
        if chat_id not in self.chat_tasks:
            self.chat_tasks[chat_id] = []
        self.chat_tasks[chat_id].append(task.task_id)

        return task

    def create_task_dedicated_chat(self, parent_chat_id: str, task: SearchTask, username: str, pipeline_model_id: str) -> str:
        """
        Creates a dedicated chat for a specific search task.

        Args:
            parent_chat_id: The original chat ID where the task was created
            task: The search task object
            username: The name of the user
            pipeline_model_id: The model ID of the pipeline that initiated the task

        Returns:
            The new chat ID
        """
        # Create a descriptive title for the new chat
        search_terms_str = ", ".join(task.search_terms)
        title = f"🔍 {search_terms_str[:50]}{'...' if len(search_terms_str) > 50 else ''}"

        # Create the new chat with the pipeline model ID
        new_chat_data = create_new_chat(model=pipeline_model_id, title=title)
        new_chat_id = new_chat_data["id"]

        # Add chat data to our local cache
        if new_chat_id not in CHATS:
            CHATS[new_chat_id] = new_chat_data["chat"]

        # Associate the task with the new chat
        self.task_chats[task.task_id] = new_chat_id

        if new_chat_id not in self.chat_tasks:
            self.chat_tasks[new_chat_id] = []
        self.chat_tasks[new_chat_id].append(task.task_id)

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
        try:
            # Create a message that appears to be from the assistant
            chat_data = CHATS[new_chat_id]

            # Create a message ID for the system message
            message_id = str(uuid.uuid4())

            # Create a system message object
            now_secs = int(time.time())
            system_msg = {
                "id": message_id,
                "parentId": None,
                "childrenIds": [],
                "role": "assistant",
                "content": intro_message,
                "model": pipeline_model_id,
                "modelIdx": 0,
                "userContext": None,
                "timestamp": now_secs
            }

            # Add to chat history
            chat_data["history"]["messages"][message_id] = system_msg
            chat_data["history"]["currentId"] = message_id

            # Add to messages list
            chat_data["messages"].append(system_msg)

            # Update the chat on the server
            update_chat_on_server(new_chat_id, chat_data)

            # Cache the message locally
            self.cache_message(new_chat_id, "assistant", intro_message)
        except Exception as e:
            print(f"Failed to send initial message to dedicated chat: {str(e)}")

        # Start the update thread for this chat
        if new_chat_id not in self.update_queues:
            self._start_update_thread(new_chat_id)

        return new_chat_id

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
                    # Send the update using the improved send_message_to_chat method
                    self.send_message_to_chat(chat_id, update)
                except Exception as e:
                    print(f"Failed to send update: {str(e)}")
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
                print(f"Chat {chat_id} not found in cache. Creating a new entry.")
                # Create a minimal chat structure - this would need to be enhanced in a real implementation
                # to properly fetch the existing chat
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
                "model": self.llm,
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

            # Cache the message
            self.cache_message(chat_id, "assistant", message)
        except Exception as e:
            print(f"Failed to send message to chat {chat_id}: {str(e)}")

    def cancel_task(self, chat_id: str, task_id: str) -> bool:
        """
        Cancel a specific task.

        Args:
            chat_id: The chat ID
            task_id: The task ID

        Returns:
            True if the task was cancelled, False otherwise
        """
        success = self.search_manager.cancel_task(chat_id, task_id)

        if success:
            # If there's a dedicated chat for this task, notify it as well
            dedicated_chat_id = self.task_chats.get(task_id)
            if dedicated_chat_id:
                self.send_message_to_chat(
                    dedicated_chat_id,
                    f"🛑 Task {task_id} has been cancelled by the user."
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
        count = self.search_manager.cancel_all_tasks(chat_id)

        # Notify all dedicated chats for the cancelled tasks
        for task_id in self.chat_tasks.get(chat_id, []):
            dedicated_chat_id = self.task_chats.get(task_id)
            if dedicated_chat_id:
                self.send_message_to_chat(
                    dedicated_chat_id,
                    f"🛑 Task {task_id} has been cancelled by the user from another chat."
                )

        return count

    def get_task(self, chat_id: str, task_id: str) -> Optional[SearchTask]:
        """
        Get a task by its ID.

        Args:
            chat_id: The chat ID
            task_id: The task ID

        Returns:
            The task if found, None otherwise
        """
        return self.search_manager.get_task(chat_id, task_id)

    def get_tasks_for_chat(self, chat_id: str) -> List[Dict[str, Any]]:
        """
        Get all tasks associated with a chat.

        Args:
            chat_id: The chat ID

        Returns:
            List of tasks
        """
        tasks = []
        for task_id in self.chat_tasks.get(chat_id, []):
            task = self.search_manager.get_task(chat_id, task_id)
            if task:
                tasks.append({
                    "task_id": task.task_id,
                    "search_terms": task.search_terms,
                    "status": task.status,
                    "duration": task.duration(),
                    "results_count": len(task.results) if hasattr(task, 'results') else 0,
                    "patterns_count": len(task.discovered_patterns) if hasattr(task, 'discovered_patterns') else 0
                })

        return tasks

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

    def schedule_task_execution(self, task: SearchTask, dedicated_chat_id: str):
        """
        Schedule the execution of a search task.

        Args:
            task: The search task
            dedicated_chat_id: The dedicated chat ID for detailed updates
        """
        if not self._event_loop_started:
            self._create_event_loop()

        self._event_loop.call_soon_threadsafe(
            asyncio.create_task,
            self._run_task_with_updates(task, dedicated_chat_id)
        )

    async def _run_task_with_updates(self, task: SearchTask, dedicated_chat_id: str):
        """
        Run a search task and send updates to the dedicated chat.

        Args:
            task: The search task
            dedicated_chat_id: The dedicated chat ID
        """
        try:
            # Initial messages are already sent by create_task_dedicated_chat

            # Run the task, passing self as task_manager for better message handling
            await self.search_manager.run_task(
                task=task,
                task_manager=self,  # Pass self as the task_manager
                dedicated_chat_id=dedicated_chat_id
            )

            # Send updates based on the task status
            if task.status == TaskStatus.COMPLETED:
                await asyncio.sleep(2)  # Delay before sending the summary

                # Generate a results summary using the LLM
                task_dict = {
                    "task_id": task.task_id,
                    "search_terms": task.search_terms,
                    "status": task.status,
                    "duration": task.duration(),
                    "results": task.results,
                    "discovered_patterns": task.discovered_patterns
                }

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

                # Send detailed summary to the dedicated chat
                self.send_message_to_chat(dedicated_chat_id, summary)

            elif task.status == TaskStatus.CANCELLED:
                self.send_message_to_chat(
                    dedicated_chat_id,
                    f"🛑 This search task was cancelled after {task.duration():.1f}s"
                )

            elif task.status == TaskStatus.FAILED:
                error_msg = task.errors[0] if task.errors else "Unknown error"
                self.send_message_to_chat(
                    dedicated_chat_id,
                    f"❌ Search failed after {task.duration():.1f}s: {error_msg}"
                )

        except asyncio.CancelledError:
            task.cancel()
            self.send_message_to_chat(
                dedicated_chat_id,
                f"🛑 Search task cancelled: {task.task_id}"
            )

        except Exception as e:
            error_msg = f"❌ Search failed: {str(e)}"
            if hasattr(task, 'fail'):
                task.fail(error_msg)
            self.send_message_to_chat(dedicated_chat_id, error_msg)
