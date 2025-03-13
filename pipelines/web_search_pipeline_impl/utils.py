import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, List, Optional, Any

from integration.net.www.chrome.chrome_surfer import search_web
from knowledge.graph.kg_models import KGNode
from knowledge.reasoning.engine.re_models import DataNode


class TaskStatus(str, Enum):
    """Enum for search task statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchTask(DataNode):
    """
    Represents a search task with associated queries, states and results.
    Extends DataNode from the CO-FORM Recursive Reasoning Framework.
    """
    task_id: str
    chat_id: str
    status: TaskStatus = TaskStatus.PENDING
    search_terms: List[str]
    semantic_patterns: Optional[List[str]] = None
    instructions: Optional[str] = None
    results: List[str] = []
    discovered_patterns: List[str] = []
    errors: List[str] = []
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    is_transient: bool = False
    created_at: float

    def __init__(self, **kwargs):
        if "task_id" not in kwargs:
            kwargs["task_id"] = str(uuid.uuid4())

        if "created_at" not in kwargs:
            kwargs["created_at"] = time.time()

        super().__init__(**kwargs)

    def start(self):
        """Start the search task and update its status"""
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        return self

    def complete(self, results: List[str], patterns: List[str]):
        """Mark the task as completed with results"""
        self.status = TaskStatus.COMPLETED
        self.end_time = time.time()
        self.results = results
        self.discovered_patterns = patterns
        return self

    def fail(self, error: str):
        """Mark the task as failed with error message"""
        self.status = TaskStatus.FAILED
        self.end_time = time.time()
        self.errors.append(error)
        return self

    def cancel(self):
        """Mark the task as cancelled"""
        self.status = TaskStatus.CANCELLED
        self.end_time = time.time()
        return self

    def duration(self) -> float:
        """Calculate the duration of the task in seconds"""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task to a dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "chat_id": self.chat_id,
            "status": self.status,
            "search_terms": self.search_terms,
            "semantic_patterns": self.semantic_patterns,
            "instructions": self.instructions,
            "results_count": len(self.results),
            "patterns_count": len(self.discovered_patterns),
            "errors": self.errors,
            "duration": self.duration(),
            "is_complete": self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        }

    def get_brief_summary(self) -> str:
        """Get a brief summary of the task's status and progress"""
        if self.status == TaskStatus.PENDING:
            return f"🕒 Task {self.task_id} is pending"

        elif self.status == TaskStatus.RUNNING:
            duration = self.duration()
            return f"🔍 Task {self.task_id} is running for {duration:.1f}s"

        elif self.status == TaskStatus.COMPLETED:
            return f"✅ Task {self.task_id} completed with {len(self.results)} results in {self.duration():.1f}s"

        elif self.status == TaskStatus.FAILED:
            error = self.errors[0] if self.errors else "Unknown error"
            return f"❌ Task {self.task_id} failed: {error}"

        elif self.status == TaskStatus.CANCELLED:
            return f"🛑 Task {self.task_id} was cancelled"

        return f"Task {self.task_id}: {self.status}"


class UserSession(KGNode):
    """
    Represents a user's chat session with associated tasks and state.
    Extends KGNode from the Knowledge Graph module.
    """
    chat_id: str
    user_id: str
    tasks: Dict[str, SearchTask] = {}
    last_update_time: float

    def __init__(self, **kwargs):
        if "last_update_time" not in kwargs:
            kwargs["last_update_time"] = time.time()
        super().__init__(**kwargs)

    def add_task(self, task: SearchTask) -> SearchTask:
        """Add a search task to this session"""
        self.tasks[task.task_id] = task
        self.last_update_time = time.time()
        return task

    def get_task(self, task_id: str) -> Optional[SearchTask]:
        """Get a task by its ID"""
        return self.tasks.get(task_id)

    def get_active_tasks(self) -> List[SearchTask]:
        """Get all active (non-completed) tasks"""
        return [t for t in self.tasks.values() if t.status in [TaskStatus.PENDING, TaskStatus.RUNNING]]

    def cancel_all_tasks(self) -> int:
        """Cancel all pending and running tasks. Returns the count of cancelled tasks."""
        active_tasks = self.get_active_tasks()
        count = 0

        for task in active_tasks:
            task.cancel()
            count += 1

        if count > 0:
            self.last_update_time = time.time()

        return count

    def get_recent_tasks(self, limit: int = 5) -> List[SearchTask]:
        """Get the most recent tasks"""
        return sorted(
            self.tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )[:limit]


class SearchManager:
    """
    Manages search tasks, user sessions, and execution of searches.
    """

    def __init__(self, max_workers: int = 8):
        self.sessions: Dict[str, UserSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.RLock()
        self.max_workers = max_workers

    async def initialize(self):
        """Initialize the search manager"""
        # Any async initialization logic can go here
        pass

    async def shutdown(self):
        """Shutdown the search manager and cleanup resources"""
        self.executor.shutdown(wait=False)

    def _get_or_create_session(self, chat_id: str, user_id: str = "unknown") -> UserSession:
        """Get or create a session for the user"""
        with self.lock:
            if chat_id not in self.sessions:
                session = UserSession(
                    uid=f"session-{chat_id}",
                    chat_id=chat_id,
                    user_id=user_id
                )
                self.sessions[chat_id] = session
            return self.sessions[chat_id]

    def create_task(self, chat_id: str, search_terms: List[str],
                    semantic_patterns: Optional[List[str]] = None,
                    instructions: Optional[str] = None,
                    user_id: str = "unknown") -> SearchTask:
        """Create a new search task"""
        session = self._get_or_create_session(chat_id, user_id)

        task = SearchTask(
            chat_id=chat_id,
            search_terms=search_terms,
            semantic_patterns=semantic_patterns,
            instructions=instructions
        )

        session.add_task(task)
        return task

    async def run_task(self, task: SearchTask, task_manager=None, dedicated_chat_id: str = None):
        """
        Run a search task and update the user with progress.
        Performs individual searches for each search term.

        Args:
            task: The search task object
            task_manager: Optional TaskManager instance for better message handling
            dedicated_chat_id: Optional dedicated chat ID for sending detailed updates
        """
        try:
            # Mark task as running
            task.start()

            # Determine how to send updates based on available components
            async def send_update(chat_id, message):
                if task_manager:
                    # Use TaskManager's methods if available
                    task_manager.send_message_to_chat(chat_id, message)
                else:
                    # Fallback to our internal method
                    await self._send_update(chat_id, message)

            # Send initial updates
            await send_update(task.chat_id, f"### 🔍 Starting search for: {', '.join(task.search_terms)}")

            # If we have a dedicated chat, send more detailed updates there
            if dedicated_chat_id:
                await send_update(dedicated_chat_id, f"### 🔍 Starting search operation with {len(task.search_terms)} search terms.")

            # Prepare containers for aggregate results
            all_results = []
            all_patterns = []

            # Execute searches in a thread pool to avoid blocking - one search per term
            loop = asyncio.get_running_loop()

            for i, term in enumerate(task.search_terms, 1):
                if task.status == TaskStatus.CANCELLED:
                    break

                # Update on progress
                term_progress_msg = f"### 🔍 Processing search term {i}/{len(task.search_terms)}: '{term}'"
                if dedicated_chat_id:
                    await send_update(dedicated_chat_id, term_progress_msg)

                # Execute the individual search
                try:
                    results, patterns = await loop.run_in_executor(
                        self.executor,
                        lambda: search_web(
                            search_terms=[term],  # Just the current term
                            semantic_patterns=task.semantic_patterns,
                            instructions=task.instructions,
                            max_workers=self.max_workers,
                            transient=task.is_transient
                        )
                    )

                    # Add to the aggregate results
                    all_results.extend(results)
                    all_patterns.extend(patterns)

                    # Send per-term update to dedicated chat
                    if dedicated_chat_id:
                        term_summary = f"### ✅ Search for '{term}' completed\n"
                        term_summary += f"Found {len(results)} results and {len(patterns)} patterns"
                        await send_update(dedicated_chat_id, term_summary)

                        # If we have results, show a sample
                        if results:
                            sample_results = f"Sample results for '{term}':\n"
                            for r in results[:3]:  # Show only 3 sample results
                                sample_results += f"- {r}\n"
                            if len(results) > 3:
                                sample_results += f"...and {len(results) - 3} more."
                            await send_update(dedicated_chat_id, sample_results)

                except Exception as e:
                    error_msg = f"### ❌ Error searching for term '{term}': {str(e)}"
                    if dedicated_chat_id:
                        await send_update(dedicated_chat_id, error_msg)
                    # Continue with other terms despite the error

            # Mark task as completed with all aggregated results
            if task.status != TaskStatus.CANCELLED:
                task.complete(all_results, all_patterns)

                # Send completion update
                summary = f"### ✅ Search completed in {task.duration():.1f}s\n"
                summary += f"Found {len(task.results)} total results and {len(task.discovered_patterns)} patterns"
                await send_update(task.chat_id, summary)

                if dedicated_chat_id:
                    detailed_summary = f"# Search Complete\n\n"
                    detailed_summary += f"### ✅ All {len(task.search_terms)} search terms processed in {task.duration():.1f}s\n\n"
                    detailed_summary += f"Total results: {len(task.results)}\n"
                    detailed_summary += f"Total patterns discovered: {len(task.discovered_patterns)}\n\n"
                    detailed_summary += "A detailed summary will be generated shortly."
                    await send_update(dedicated_chat_id, detailed_summary)

        except asyncio.CancelledError:
            task.cancel()
            cancel_msg = f"### 🛑 Search task cancelled: {task.task_id}"
            await self._send_update(task.chat_id, cancel_msg)
            if dedicated_chat_id:
                await self._send_update(dedicated_chat_id, cancel_msg)

        except Exception as e:
            error_msg = f"### ❌ Search failed: {str(e)}"
            task.fail(error_msg)
            await self._send_update(task.chat_id, error_msg)
            if dedicated_chat_id:
                await self._send_update(dedicated_chat_id, error_msg)

    async def _send_update(self, chat_id: str, message: str):
        """
        Send an update message to the user's chat.
        This now defers to the TaskManager's methods which handle chat updates properly.

        Args:
            chat_id: The chat ID
            message: The message to send
        """
        try:
            # Use asyncio to avoid blocking
            loop = asyncio.get_running_loop()

            # This will now use the improved mechanism through the TaskManager
            # We need to get a reference to the TaskManager instance
            # This requires a refactor where we pass the TaskManager as a dependency
            # or make it globally accessible

            # For now, we'll use the direct update_chat_on_server approach with proper message structure
            await loop.run_in_executor(
                None,
                lambda: self._direct_send_message(chat_id, message)
            )
        except Exception as e:
            print(f"Failed to send update: {str(e)}")

    def _direct_send_message(self, chat_id: str, message: str):
        """
        Directly send a message to a chat. This is a fallback method when TaskManager is not available.

        Args:
            chat_id: The chat ID
            message: The message content
        """
        try:
            from integration.net.open_webui.api import update_chat_on_server, CHATS

            # Try to get existing chat data
            if chat_id not in CHATS:
                print(f"Chat {chat_id} not found in CHATS. Creating a minimal entry.")
                # Create a minimal chat structure
                CHATS[chat_id] = {
                    "id": chat_id,
                    "models": ["unknown-model"],
                    "history": {
                        "messages": {},
                        "currentId": None
                    },
                    "messages": []
                }

            chat_data = CHATS[chat_id]

            # Create a message ID for the new message
            import uuid
            message_id = str(uuid.uuid4())

            # Create a message object
            import time
            now_secs = int(time.time())
            msg_obj = {
                "id": message_id,
                "parentId": chat_data["history"]["currentId"],
                "childrenIds": [],
                "role": "assistant",
                "content": message,
                "model": "unknown-model",
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
        except Exception as e:
            print(f"Failed to send direct message to chat {chat_id}: {str(e)}")

    def cancel_task(self, chat_id: str, task_id: str) -> bool:
        """Cancel a specific task"""
        session = self.sessions.get(chat_id)
        if not session:
            return False

        task = session.get_task(task_id)
        if not task or task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            return False

        task.cancel()
        return True

    def cancel_all_tasks(self, chat_id: str) -> int:
        """Cancel all active tasks for a chat. Returns the count of cancelled tasks."""
        session = self.sessions.get(chat_id)
        if not session:
            return 0

        return session.cancel_all_tasks()

    def get_task(self, chat_id: str, task_id: str) -> Optional[SearchTask]:
        """Get a specific task"""
        session = self.sessions.get(chat_id)
        if not session:
            return None

        return session.get_task(task_id)

    def get_results_summary(self, chat_id: str, task_id: Optional[str] = None) -> str:
        """Generate a summary of search results for a specific task or all completed tasks"""
        session = self.sessions.get(chat_id)
        if not session:
            return "No active search session found."

        if task_id:
            # Summarize specific task
            task = session.get_task(task_id)
            if not task:
                return f"No task found with ID: {task_id}"

            if task.status != TaskStatus.COMPLETED:
                return f"Task status: {task.status}. No results available yet."

            summary = f"Search results for: {', '.join(task.search_terms)}\n\n"

            # Include discovered patterns
            if task.discovered_patterns:
                summary += "Discovered patterns:\n"
                for pattern in task.discovered_patterns[:5]:  # Limit to top 5
                    summary += f"- {pattern}\n"
                summary += "\n"

            # Include top results
            if task.results:
                summary += "Top results:\n"
                for result in task.results[:10]:  # Limit to top 10
                    summary += f"- {result}\n"
            else:
                summary += "No results found."

            return summary
        else:
            # Summarize all completed tasks
            completed_tasks = [t for t in session.tasks.values() if t.status == TaskStatus.COMPLETED]
            if not completed_tasks:
                return "No completed search tasks found."

            summary = f"Summary of {len(completed_tasks)} completed searches:\n\n"

            for i, task in enumerate(completed_tasks, 1):
                summary += f"{i}. Query: {', '.join(task.search_terms)}\n"
                summary += f"   Results: {len(task.results)} | Patterns: {len(task.discovered_patterns)}\n"
                summary += f"   ID: {task.task_id}\n\n"

            summary += "Use 'show results for [task ID]' to see detailed results."
            return summary

    def get_status_summary(self, chat_id: str) -> str:
        """Get a summary of all tasks for a chat"""
        session = self.sessions.get(chat_id)
        if not session:
            return "No active search session found."

        tasks = session.tasks.values()
        if not tasks:
            return "No search tasks found."

        active_tasks = [t for t in tasks if t.status in [TaskStatus.PENDING, TaskStatus.RUNNING]]
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in tasks if t.status in [TaskStatus.FAILED, TaskStatus.CANCELLED]]

        summary = "Current search status:\n\n"

        if active_tasks:
            summary += f"Active tasks ({len(active_tasks)}):\n"
            for task in active_tasks:
                summary += f"- {task.get_brief_summary()}\n"
            summary += "\n"

        if completed_tasks:
            summary += f"Completed tasks ({len(completed_tasks)}):\n"
            for task in completed_tasks[:3]:  # Show only the 3 most recent
                summary += f"- {task.get_brief_summary()}\n"
            if len(completed_tasks) > 3:
                summary += f"  ... and {len(completed_tasks) - 3} more\n"
            summary += "\n"

        if failed_tasks:
            summary += f"Failed/cancelled tasks ({len(failed_tasks)}):\n"
            for task in failed_tasks[:3]:  # Show only the 3 most recent
                summary += f"- {task.get_brief_summary()}\n"
            if len(failed_tasks) > 3:
                summary += f"  ... and {len(failed_tasks) - 3} more\n"

        return summary
