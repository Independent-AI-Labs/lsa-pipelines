"""
Web Search Pipeline with Knowledge Graph integration.

This module implements the main pipeline that processes user messages,
detects intents, and manages search tasks using the Knowledge Graph and
Recursive Reasoning Framework.
"""

import logging
import uuid
from typing import Dict, List, Any, Union, Generator, Iterator, Optional

from integration.data.config import DEFAULT_LLM, KGML_MODEL
from integration.net.ollama.ollama_api import prompt_model
from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants import (
    NodeType, TaskStatus
)
from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_models import (
    UserMessageEvent, ChatMessage, WebSearchFunction,
    PatternExtractionFunction
)
from integration.pipelines.pipelines.web_search_pipeline_impl.manage.kg_manager import KnowledgeGraphManager
from integration.pipelines.pipelines.web_search_pipeline_impl.util.kgml_handler import KGMLHandler
from knowledge.reasoning.dsl.execution.kgml_executor import KGMLExecutor
from knowledge.reasoning.dsl.kgml_system_prompt_generator import generate_kgml_system_prompt


class Pipeline:
    """
    Web Search Pipeline with Knowledge Graph integration.

    This pipeline uses a Knowledge Graph and Recursive Reasoning Framework
    to intelligently analyze user queries, maintain context, and manage
    search tasks for improved conversational capabilities.
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.name = "Web Search Assistant"
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize managers
        self.kg_manager = KnowledgeGraphManager()
        self.kgml_manager = KGMLHandler()

        # Initialize predefined system function nodes
        self._initialize_system_functions()

        # Model configuration
        self.default_model = DEFAULT_LLM

    async def on_startup(self):
        """Initialize resources when the server starts."""
        self.logger.info(f"on_startup:{__name__}")
        await self.kg_manager.initialize()

    async def on_shutdown(self):
        """Clean up resources when the server stops."""
        self.logger.info(f"on_shutdown:{__name__}")
        await self.kg_manager.shutdown()

    def _initialize_system_functions(self):
        """Initialize predefined system function nodes in the knowledge graph."""
        try:
            # Create WebSearchFunction if not exists
            web_search_nodes = self.kg_manager.graph.query_nodes(type=NodeType.WEB_SEARCH)
            if not web_search_nodes:
                web_search = WebSearchFunction(
                    uid=f"function-websearch",
                    type=NodeType.WEB_SEARCH,
                )
                self.kg_manager.graph.add_node(web_search)
                self.logger.info("Created WebSearchFunction node")

            # Create PatternExtractionFunction if not exists
            pattern_nodes = self.kg_manager.graph.query_nodes(type=NodeType.PATTERN_EXTRACTION)
            if not pattern_nodes:
                pattern_function = PatternExtractionFunction(
                    uid=f"function-patternextraction",
                    type=NodeType.PATTERN_EXTRACTION,
                )
                self.kg_manager.graph.add_node(pattern_function)
                self.logger.info("Created PatternExtractionFunction node")

        except Exception as e:
            self.logger.error(f"Error initializing system functions: {str(e)}")

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

    def _process_user_message(self, chat_id: str, user_id: str, user_message: str, model: str) -> Dict[str, Any]:
        """
        Process a user message using the Knowledge Graph.

        Args:
            chat_id: The chat ID
            user_id: The user ID
            user_message: The user's message
            model: The model ID to use for LLM prompting

        Returns:
            Dictionary with processing results
        """
        try:
            # Add the user message to the knowledge graph
            message_id = str(uuid.uuid4())

            # Create chat message node
            chat_message = ChatMessage(
                message_id=message_id,
                chat_id=chat_id,
                role="user",
                content=user_message,
                metadata={"user_id": user_id}
            )
            self.kg_manager.graph.add_node(chat_message)

            # Create user message event
            message_event = UserMessageEvent(
                message_id=message_id,
                chat_id=chat_id,
                user_id=user_id,
                content=user_message
            )
            self.kg_manager.graph.add_node(message_event)

            # Update conversation context
            context = self.kg_manager.get_or_create_context(chat_id)
            context.update_with_message(chat_message)
            self.kg_manager.graph.update_node(context.uid, context.model_dump())

            # Get relevant KG subset for KGML processing
            focal_nodes = [message_event.uid, context.uid]
            active_configs = []

            # Add recent active search configs to focal nodes
            for config_id in context.active_configs:
                config = self.kg_manager.get_search_config(config_id)
                if config and config.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    active_configs.append(config)
                    focal_nodes.append(config.uid)

            # If no active configs, get most recent completed configs
            if not active_configs:
                completed_configs = self.kg_manager.get_recent_completed_configs(chat_id, limit=1)
                for config in completed_configs:
                    focal_nodes.append(config.uid)

            # Serialize relevant KG subset with current node marker
            kg_subset = self.kgml_manager.serialize_kg_subset(
                self.kg_manager.graph,
                focal_nodes=focal_nodes,
                chat_id=chat_id,
                current_node_id=message_event.uid  # Mark the current message event as the focal point
            )

            # Process with KGML agent
            self.logger.info(f"Processing message with KGML agent")
            kgml_response = self._process_with_kgml_agent(kg_subset, model=model)

            # Execute KGML response directly with EnhancedKGMLExecutor
            results = self._execute_kgml_program(kgml_response, chat_id=chat_id)

            # Extract assistant's response from results
            assistant_response = self._extract_assistant_response(results, chat_id)

            # Add the assistant's response to the knowledge graph
            if assistant_response:
                assistant_message = ChatMessage(
                    message_id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    role="assistant",
                    content=assistant_response
                )
                self.kg_manager.graph.add_node(assistant_message)

                # Update context with assistant message
                context.update_with_message(assistant_message)
                self.kg_manager.graph.update_node(context.uid, context.model_dump())

            return {
                "message_id": message_id,
                "response": assistant_response or "I'm sorry, I couldn't process your request properly."
            }

        except Exception as e:
            self.logger.error(f"Error processing user message: {str(e)}")
            return {
                "message_id": str(uuid.uuid4()),
                "response": f"I encountered an error while processing your request: {str(e)}"
            }

    def _process_with_kgml_agent(self, kg_subset: str, model: str = None) -> str:
        """
        Process a Knowledge Graph subset using the KGML agent.

        Args:
            kg_subset: Serialized KG subset in KGML format
            model: Optional model ID to use

        Returns:
            KGML response from the agent
        """
        try:
            # Use the generate_kgml_system_prompt function instead of loading from file
            system_prompt = generate_kgml_system_prompt()
        except Exception as e:
            self.logger.error(f"Error generating KGML system prompt: {str(e)}")
            # Use a default system prompt if generation fails
            system_prompt = "You are a KGML Reasoning Agent. Process the Knowledge Graph and respond with valid KGML operations."

        # Use Ollama API to process the KGML
        model_to_use = model or KGML_MODEL

        response = prompt_model(
            message=kg_subset,
            model=model_to_use,
            system_prompt=system_prompt
        )

        return response.replace("```", "").strip()

    def _execute_kgml_program(self, kgml_text: str, chat_id: str) -> List[Dict[str, Any]]:
        """
        Execute a KGML program directly using the EnhancedKGMLExecutor.

        Args:
            kgml_text: The KGML program text
            chat_id: The chat ID for context

        Returns:
            List of operation results
        """
        try:
            # Initialize the executor
            executor = KGMLExecutor(self.kg_manager.graph)

            # Execute the KGML program
            context = executor.execute(kgml_text)

            # Process the results from execution context
            results = []
            for entry in context.execution_log:
                result = {
                    "operation": {
                        "command": entry["command_type"],
                        "entity_type": entry["details"].get("entity_type"),
                        "entity_id": entry["details"].get("uid"),
                        "instruction": entry["details"].get("instruction")
                    },
                    "result": entry["result"],
                    "success": entry["success"]
                }
                results.append(result)

            return results
        except Exception as e:
            self.logger.error(f"Error executing KGML program: {str(e)}")
            return [{
                "operation": {"error": "execution_failed"},
                "error": str(e)
            }]

    def _extract_assistant_response(self, operation_results: List[Dict[str, Any]], chat_id: str) -> Optional[str]:
        """
        Extract the assistant's response from operation results.

        Args:
            operation_results: Results from executing KGML operations
            chat_id: The chat ID

        Returns:
            The assistant's response text, or None if no response was generated
        """
        # Look for response in operation results
        for result in operation_results:
            op_result = result.get("result", {})
            if op_result.get("status") == "created" and op_result.get("node_type") == NodeType.CHAT_MESSAGE:
                # Find the message node
                try:
                    node_id = op_result.get("node_id")
                    node = self.kg_manager.graph.get_node(node_id)
                    if node and hasattr(node, "role") and node.role == "assistant":
                        return node.content
                except Exception as e:
                    self.logger.error(f"Error retrieving assistant message: {str(e)}")

        # If no response was found in operations, check for direct message content
        for result in operation_results:
            if "direct_response" in result.get("result", {}):
                return result["result"]["direct_response"]

        # If still no response, try to find the most recent generated response in the KG
        try:
            recent_messages = self.kg_manager.get_recent_messages(chat_id, role="assistant", limit=1)
            if recent_messages:
                return recent_messages[0].content
        except Exception as e:
            self.logger.error(f"Error retrieving recent assistant message: {str(e)}")

        return None

    def _finish_response(self, response: str) -> Union[str, Generator, Iterator]:
        """
        Finalize and return a response.

        Args:
            response: The response message

        Returns:
            The yielded response
        """
        yield "</think>"
        yield response
        return

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

        # Ignore autocomplete requests.
        if user_message.startswith("### Task:"):
            return ""

        # Extract chat info
        chat_info = self._extract_chat_info(body)
        chat_id = chat_info["chat_id"]
        user_id = chat_info["user_id"]

        yield "<think>"
        yield " "

        # Store the pipeline model_id for dedicated chats and use default_model for LLM operations
        pipeline_model_id = model_id  # This is the model that represents the pipeline itself
        llm = self.default_model  # This is what we'll use for actual LLM operations

        try:
            # Process the user message
            result = self._process_user_message(
                chat_id=chat_id,
                user_id=user_id,
                user_message=user_message,
                model=llm
            )

            # Return the response
            if "response" in result:
                yield from self._finish_response(result["response"])
            else:
                yield from self._finish_response(
                    "I'm sorry, I encountered an error processing your request. Please try again."
                )

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            error_response = f"I encountered an error while processing your request: {str(e)}"
            yield from self._finish_response(error_response)
