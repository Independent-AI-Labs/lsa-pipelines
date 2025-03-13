"""
KGML Manager for the Web Search Assistant.

This module handles the serialization of Knowledge Graph elements to KGML format,
as well as relevance scoring for nodes and links to determine which should be
included in prompts.
"""

from typing import Dict, List, Any, Optional, Tuple

from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants import (
    MAX_FIELD_LENGTH, MAX_KG_NODES
)
from knowledge.graph.kg_models import KGNode, KGEdge, KnowledgeGraph


class KGMLHandler:
    """
    Manager for KGML serialization and relevance scoring.

    Handles the conversion between Knowledge Graph objects and KGML text format,
    as well as determining which nodes and links are most relevant for inclusion
    in LLM prompts.
    """

    def __init__(self):
        """Initialize the KGML Manager."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def serialize_kg_subset(self,
                            graph: KnowledgeGraph,
                            focal_nodes: List[str] = None,
                            chat_id: Optional[str] = None,
                            current_node_id: Optional[str] = None,
                            max_nodes: int = MAX_KG_NODES) -> str:
        """
        Serialize a relevant subset of the Knowledge Graph to KGML format.

        Args:
            graph: The Knowledge Graph to serialize
            focal_nodes: Optional list of node UIDs to focus on
            chat_id: Optional chat ID to filter nodes by
            current_node_id: Optional ID of the current node being processed
            max_nodes: Maximum number of nodes to include

        Returns:
            String containing KGML representation of the relevant KG subset
        """
        # Identify relevant nodes
        nodes, links = self._get_relevant_subset(
            graph,
            focal_nodes=focal_nodes,
            chat_id=chat_id,
            max_nodes=max_nodes
        )

        # Generate KGML
        kgml = "KG►\n"

        # Add current node marker comment if provided
        if current_node_id:
            kgml += f"# CURRENT_NODE: {current_node_id}\n"

        # Add nodes
        for node in nodes:
            # Mark the current node if it matches
            if current_node_id and node.uid == current_node_id:
                kgml += "# ↓↓↓ CURRENT NODE ↓↓↓\n"
                kgml += self._node_to_kgml(node)
                kgml += "# ↑↑↑ CURRENT NODE ↑↑↑\n"
            else:
                kgml += self._node_to_kgml(node)

        # Add links
        for link in links:
            # Mark links connected to the current node
            if current_node_id and (link.source_uid == current_node_id or link.target_uid == current_node_id):
                kgml += "# ↓↓↓ CURRENT NODE LINK ↓↓↓\n"
                kgml += self._link_to_kgml(link)
                kgml += "# ↑↑↑ CURRENT NODE LINK ↑↑↑\n"
            else:
                kgml += self._link_to_kgml(link)

        kgml += "◄\n"

        return kgml

    def _get_relevant_subset(self,
                             graph: KnowledgeGraph,
                             focal_nodes: List[str] = None,
                             chat_id: Optional[str] = None,
                             max_nodes: int = MAX_KG_NODES) -> Tuple[List[KGNode], List[KGEdge]]:
        """
        Get the most relevant subset of nodes and links from the graph.

        Uses relevance scoring to determine which nodes to include in the
        serialized KG, starting from focal nodes and traversing the graph.

        Args:
            graph: The Knowledge Graph
            focal_nodes: Optional list of node UIDs to focus on
            chat_id: Optional chat ID to filter nodes by
            max_nodes: Maximum number of nodes to include

        Returns:
            Tuple of (nodes, links) to include in the serialized KG
        """
        all_nodes = []

        # If chat_id is provided, get all nodes associated with that chat
        if chat_id:
            # Query for nodes with this chat_id
            try:
                chat_nodes = graph.query_nodes(chat_id=chat_id)
                all_nodes.extend(chat_nodes)
            except Exception as e:
                self.logger.error(f"Error querying nodes by chat_id: {str(e)}")

        # If focal_nodes are provided, add them first
        focal_node_objs = []
        if focal_nodes:
            for uid in focal_nodes:
                try:
                    node = graph.get_node(uid)
                    if node and node not in all_nodes:
                        all_nodes.append(node)
                        focal_node_objs.append(node)
                except Exception as e:
                    self.logger.error(f"Error getting focal node {uid}: {str(e)}")

        # Score all nodes for relevance
        scored_nodes = [(node, self._calculate_node_relevance(node, focal_node_objs))
                        for node in all_nodes]

        # Sort by relevance score (descending)
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # Get the top N most relevant nodes
        relevant_nodes = [node for node, score in scored_nodes[:max_nodes]]
        relevant_node_uids = {node.uid for node in relevant_nodes}

        # Get links between relevant nodes
        relevant_links = []
        for source_node in relevant_nodes:
            try:
                # Get outgoing edges using the optimized method
                outgoing_edges = graph.get_outgoing_edges(source_node.uid)
                for edge in outgoing_edges:
                    if edge.target_uid in relevant_node_uids:
                        relevant_links.append(edge)
            except Exception as e:
                self.logger.error(f"Error getting edges for node {source_node.uid}: {str(e)}")

        return relevant_nodes, relevant_links

    def _calculate_node_relevance(self, node: KGNode, focal_nodes: List[KGNode] = None) -> float:
        """
        Calculate the relevance score for a node.

        Higher scores indicate higher relevance for inclusion in prompts.

        Args:
            node: The node to score
            focal_nodes: Optional list of focal nodes to compare against

        Returns:
            Relevance score (higher is more relevant)
        """
        # [Keep existing implementation...]
        score = 0.0

        # Base scores by node type
        type_scores = {
            NodeType.USER_MESSAGE: 0.9,  # User messages are very relevant
            NodeType.CHAT_MESSAGE: 0.8,  # Chat messages are relevant
            NodeType.SEARCH_CONFIG: 0.7,  # Search configs are quite relevant
            NodeType.SEARCH_RESULT: 0.6,  # Search results are moderately relevant
            NodeType.SEARCH_PATTERN: 0.7,  # Patterns are more relevant than raw results
            NodeType.CONTEXT: 0.8,  # Context is highly relevant
            NodeType.WEB_SEARCH: 0.5,  # Web search function is somewhat relevant
            NodeType.INTENT_ANALYSIS: 0.5,  # Intent analysis is somewhat relevant
            NodeType.PATTERN_EXTRACTION: 0.5,  # Pattern extraction is somewhat relevant
            NodeType.SEARCH_GOAL: 0.8,  # Goals are highly relevant
            NodeType.INTENT_OUTCOME: 0.7,  # Intent outcomes are quite relevant
        }

        # Start with base score by type
        if hasattr(node, 'type') and node.type in type_scores:
            score = type_scores[node.type]
        else:
            score = 0.3  # Default score for unknown node types

        # Adjust score based on recency
        if hasattr(node, 'timestamp'):
            # More recent nodes are more relevant
            now = datetime.now().timestamp()
            age_hours = (now - node.timestamp) / 3600
            # Exponential decay based on age
            recency_factor = max(0.1, 1.0 * (0.9 ** min(age_hours, 24)))
            score *= recency_factor

        # Adjust score based on proximity to focal nodes
        if focal_nodes:
            # Higher score if this node is directly connected to focal nodes
            for focal_node in focal_nodes:
                if hasattr(node, 'chat_id') and hasattr(focal_node, 'chat_id'):
                    if node.chat_id == focal_node.chat_id:
                        score *= 1.2  # Boost for nodes in same chat

                # Additional boost for related content
                if hasattr(node, 'config_id') and hasattr(focal_node, 'config_id'):
                    if node.config_id == focal_node.config_id:
                        score *= 1.5  # Strong boost for nodes related to same search

                # Boost if this is a result for a focal search config
                if (node.type == NodeType.SEARCH_RESULT and
                        hasattr(node, 'config_id') and
                        hasattr(focal_node, 'config_id') and
                        node.config_id == focal_node.config_id):
                    score *= 1.3

        return score

    def _truncate_value(self, value: Any, max_length: int = MAX_FIELD_LENGTH) -> str:
        """
        Truncate a value to a maximum length for serialization.

        Args:
            value: The value to truncate
            max_length: Maximum length to truncate to

        Returns:
            Truncated string representation of the value
        """
        str_value = str(value)
        if len(str_value) <= max_length:
            return str_value

        # Truncate with ellipsis
        return str_value[:max_length - 3] + "..."

    def _node_to_kgml(self, node: KGNode) -> str:
        """
        Convert a node to KGML format.

        Args:
            node: The KGNode to convert

        Returns:
            KGML string representation of the node
        """
        # Start with node identifier
        kgml = f"KGNODE► {node.uid} : "

        # Get node properties
        props = {}

        # Add type
        if hasattr(node, 'type'):
            props['type'] = node.type

        # Add common properties based on node type
        if hasattr(node, 'model_dump'):
            node_data = node.model_dump()

            # Filter out internal and redundant properties
            exclude_props = {'uid', 'meta_props', 'created_at', 'updated_at'}

            for key, value in node_data.items():
                if key not in exclude_props and value is not None:
                    props[key] = self._truncate_value(value)

        # Add meta properties if available
        if hasattr(node, 'meta_props') and node.meta_props:
            for key, value in node.meta_props.items():
                if value is not None:
                    props[f"meta.{key}"] = self._truncate_value(value)

        # Add timestamps
        if hasattr(node, 'created_at') and node.created_at:
            props['created_at'] = node.created_at
        if hasattr(node, 'updated_at') and node.updated_at:
            props['updated_at'] = node.updated_at

        # Serialize properties
        props_str = ", ".join([f'{k}="{v}"' for k, v in props.items()])
        kgml += props_str + " ◄\n"

        return kgml

    def _link_to_kgml(self, link: KGEdge) -> str:
        """
        Convert a link to KGML format.

        Args:
            link: The KGEdge to convert

        Returns:
            KGML string representation of the link
        """
        # Format: KGLINK► source_uid -> target_uid : relation="value", meta.prop="value" ◄
        kgml = f"KGLINK► {link.source_uid} -> {link.target_uid} : "

        # Get link properties
        props = {}

        # Add relation
        if hasattr(link, 'relation'):
            props['relation'] = link.relation

        # Add metadata if available
        if hasattr(link, 'meta_props') and link.meta_props:
            for key, value in link.meta_props.items():
                if value is not None:
                    props[f"meta.{key}"] = self._truncate_value(value)

        # Serialize properties
        props_str = ", ".join([f'{k}="{v}"' for k, v in props.items()])
        kgml += props_str + " ◄\n"

        return kgml

    # Replace the regex-based parsing with proper parser usage
    def parse_kgml_response(self, kgml_text: str) -> List[Dict[str, Any]]:
        """
        Parse a KGML response using the proper KGML parser.

        This method is maintained for backward compatibility but uses
        the proper parser infrastructure.

        Args:
            kgml_text: The KGML text to parse

        Returns:
            List of operation dictionaries
        """
        try:
            # Use the proper KGML parser
            tokens = tokenize(kgml_text)
            parser = Parser(tokens)
            ast = parser.parse_program()

            # Convert AST to the expected operation dictionaries format
            operations = self._ast_to_operations(ast)

            return operations
        except Exception as e:
            self.logger.error(f"Error parsing KGML: {str(e)}")
            return []

    def _ast_to_operations(self, ast) -> List[Dict[str, Any]]:
        """
        Convert a parsed AST to the operation dictionaries format.

        This is for backward compatibility with existing code that expects
        operations in dictionary format.

        Args:
            ast: The parsed AST

        Returns:
            List of operation dictionaries
        """
        operations = []

        # Process each statement in the program
        for statement in ast.statements:
            # Handle simple commands (C►, U►, D►, E►)
            if hasattr(statement, 'cmd_type') and statement.cmd_type in ['C►', 'U►', 'D►', 'E►']:
                operations.append({
                    'command': statement.cmd_type.rstrip('►'),  # Remove the trailing marker
                    'entity_type': statement.entity_type,
                    'entity_id': statement.uid,
                    'instruction': statement.instruction
                })
            # Handle conditional statements
            elif hasattr(statement, 'if_clause'):
                # Extract condition command
                condition_cmd, if_block = statement.if_clause

                operations.append({
                    'command': 'IF',
                    'condition': condition_cmd.instruction if hasattr(condition_cmd, 'instruction') else "",
                    'body': self._format_block(if_block)
                })

                # Handle elif clauses
                if hasattr(statement, 'elif_clauses') and statement.elif_clauses:
                    for i, (elif_cmd, elif_block) in enumerate(statement.elif_clauses):
                        operations.append({
                            'command': 'ELIF',
                            'condition': elif_cmd.instruction if hasattr(elif_cmd, 'instruction') else "",
                            'body': self._format_block(elif_block)
                        })

                # Handle else clause
                if hasattr(statement, 'else_clause') and statement.else_clause:
                    operations.append({
                        'command': 'ELSE',
                        'body': self._format_block(statement.else_clause)
                    })
            # Handle loop statements
            elif hasattr(statement, 'condition') and hasattr(statement, 'block'):
                operations.append({
                    'command': 'LOOP',
                    'instruction': statement.condition,
                    'body': self._format_block(statement.block)
                })
            # Handle KG blocks (could add if needed)

        return operations

    def _format_block(self, block) -> str:
        """
        Format a block of AST statements as a string.

        Args:
            block: List of AST statement nodes

        Returns:
            Formatted string representation of the block
        """
        result = ""
        for stmt in block:
            if hasattr(stmt, 'cmd_type') and stmt.cmd_type in ['C►', 'U►', 'D►', 'E►']:
                result += f"{stmt.cmd_type} {stmt.entity_type} {stmt.uid} \"{stmt.instruction}\" ◄\n"
            # Handle nested blocks if needed

        return result
        """
        KGML Manager for the Web Search Assistant.
        
        This module handles the serialization of Knowledge Graph elements to KGML format,
        as well as relevance scoring for nodes and links to determine which should be
        included in prompts.
        """


import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from integration.pipelines.pipelines.web_search_pipeline_impl.data.ws_constants import (
    MAX_FIELD_LENGTH, MAX_KG_NODES, NodeType
)
from knowledge.graph.kg_models import KGNode, KGEdge, KnowledgeGraph
from knowledge.reasoning.dsl.kgml_parser import tokenize, Parser


class KGMLHandler:
    """
    Manager for KGML serialization and relevance scoring.

    Handles the conversion between Knowledge Graph objects and KGML text format,
    as well as determining which nodes and links are most relevant for inclusion
    in LLM prompts.
    """

    def __init__(self):
        """Initialize the KGML Manager."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def serialize_kg_subset(self,
                            graph: KnowledgeGraph,
                            focal_nodes: List[str] = None,
                            chat_id: Optional[str] = None,
                            current_node_id: Optional[str] = None,
                            max_nodes: int = MAX_KG_NODES) -> str:
        """
        Serialize a relevant subset of the Knowledge Graph to KGML format.

        Args:
            graph: The Knowledge Graph to serialize
            focal_nodes: Optional list of node UIDs to focus on
            chat_id: Optional chat ID to filter nodes by
            current_node_id: Optional ID of the current node being processed
            max_nodes: Maximum number of nodes to include

        Returns:
            String containing KGML representation of the relevant KG subset
        """
        # Identify relevant nodes
        nodes, links = self._get_relevant_subset(
            graph,
            focal_nodes=focal_nodes,
            chat_id=chat_id,
            max_nodes=max_nodes
        )

        # Generate KGML
        kgml = "KG►\n"

        # Add current node marker comment if provided
        if current_node_id:
            kgml += f"# CURRENT_NODE: {current_node_id}\n"

        # Add nodes
        for node in nodes:
            # Mark the current node if it matches
            if current_node_id and node.uid == current_node_id:
                kgml += "# ↓↓↓ CURRENT NODE ↓↓↓\n"
                kgml += self._node_to_kgml(node)
                kgml += "# ↑↑↑ CURRENT NODE ↑↑↑\n"
            else:
                kgml += self._node_to_kgml(node)

        # Add links
        for link in links:
            # Mark links connected to the current node
            if current_node_id and (link.source_uid == current_node_id or link.target_uid == current_node_id):
                kgml += "# ↓↓↓ CURRENT NODE LINK ↓↓↓\n"
                kgml += self._link_to_kgml(link)
                kgml += "# ↑↑↑ CURRENT NODE LINK ↑↑↑\n"
            else:
                kgml += self._link_to_kgml(link)

        kgml += "◄\n"

        return kgml

    def _get_relevant_subset(self,
                             graph: KnowledgeGraph,
                             focal_nodes: List[str] = None,
                             chat_id: Optional[str] = None,
                             max_nodes: int = MAX_KG_NODES) -> Tuple[List[KGNode], List[KGEdge]]:
        """
        Get the most relevant subset of nodes and links from the graph.

        Uses relevance scoring to determine which nodes to include in the
        serialized KG, starting from focal nodes and traversing the graph.

        Args:
            graph: The Knowledge Graph
            focal_nodes: Optional list of node UIDs to focus on
            chat_id: Optional chat ID to filter nodes by
            max_nodes: Maximum number of nodes to include

        Returns:
            Tuple of (nodes, links) to include in the serialized KG
        """
        all_nodes = []

        # If chat_id is provided, get all nodes associated with that chat
        if chat_id:
            # Query for nodes with this chat_id
            try:
                chat_nodes = graph.query_nodes(chat_id=chat_id)
                all_nodes.extend(chat_nodes)
            except Exception as e:
                self.logger.error(f"Error querying nodes by chat_id: {str(e)}")

        # If focal_nodes are provided, add them first
        focal_node_objs = []
        if focal_nodes:
            for uid in focal_nodes:
                try:
                    node = graph.get_node(uid)
                    if node and node not in all_nodes:
                        all_nodes.append(node)
                        focal_node_objs.append(node)
                except Exception as e:
                    self.logger.error(f"Error getting focal node {uid}: {str(e)}")

        # Score all nodes for relevance
        scored_nodes = [(node, self._calculate_node_relevance(node, focal_node_objs))
                        for node in all_nodes]

        # Sort by relevance score (descending)
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # Get the top N most relevant nodes
        relevant_nodes = [node for node, score in scored_nodes[:max_nodes]]
        relevant_node_uids = {node.uid for node in relevant_nodes}

        # Get links between relevant nodes
        relevant_links = []
        for source_node in relevant_nodes:
            try:
                # Get outgoing edges using the optimized method
                outgoing_edges = graph.get_outgoing_edges(source_node.uid)
                for edge in outgoing_edges:
                    if edge.target_uid in relevant_node_uids:
                        relevant_links.append(edge)
            except Exception as e:
                self.logger.error(f"Error getting edges for node {source_node.uid}: {str(e)}")

        return relevant_nodes, relevant_links

    def _calculate_node_relevance(self, node: KGNode, focal_nodes: List[KGNode] = None) -> float:
        """
        Calculate the relevance score for a node.

        Higher scores indicate higher relevance for inclusion in prompts.

        Args:
            node: The node to score
            focal_nodes: Optional list of focal nodes to compare against

        Returns:
            Relevance score (higher is more relevant)
        """
        # [Keep existing implementation...]
        score = 0.0

        # Base scores by node type
        type_scores = {
            NodeType.USER_MESSAGE: 0.9,  # User messages are very relevant
            NodeType.CHAT_MESSAGE: 0.8,  # Chat messages are relevant
            NodeType.SEARCH_CONFIG: 0.7,  # Search configs are quite relevant
            NodeType.SEARCH_RESULT: 0.6,  # Search results are moderately relevant
            NodeType.SEARCH_PATTERN: 0.7,  # Patterns are more relevant than raw results
            NodeType.CONTEXT: 0.8,  # Context is highly relevant
            NodeType.WEB_SEARCH: 0.5,  # Web search function is somewhat relevant
            NodeType.INTENT_ANALYSIS: 0.5,  # Intent analysis is somewhat relevant
            NodeType.PATTERN_EXTRACTION: 0.5,  # Pattern extraction is somewhat relevant
            NodeType.SEARCH_GOAL: 0.8,  # Goals are highly relevant
            NodeType.INTENT_OUTCOME: 0.7,  # Intent outcomes are quite relevant
        }

        # Start with base score by type
        if hasattr(node, 'type') and node.type in type_scores:
            score = type_scores[node.type]
        else:
            score = 0.3  # Default score for unknown node types

        # Adjust score based on recency
        if hasattr(node, 'timestamp'):
            # More recent nodes are more relevant
            now = datetime.now().timestamp()
            age_hours = (now - node.timestamp) / 3600
            # Exponential decay based on age
            recency_factor = max(0.1, 1.0 * (0.9 ** min(age_hours, 24)))
            score *= recency_factor

        # Adjust score based on proximity to focal nodes
        if focal_nodes:
            # Higher score if this node is directly connected to focal nodes
            for focal_node in focal_nodes:
                if hasattr(node, 'chat_id') and hasattr(focal_node, 'chat_id'):
                    if node.chat_id == focal_node.chat_id:
                        score *= 1.2  # Boost for nodes in same chat

                # Additional boost for related content
                if hasattr(node, 'config_id') and hasattr(focal_node, 'config_id'):
                    if node.config_id == focal_node.config_id:
                        score *= 1.5  # Strong boost for nodes related to same search

                # Boost if this is a result for a focal search config
                if (node.type == NodeType.SEARCH_RESULT and
                        hasattr(node, 'config_id') and
                        hasattr(focal_node, 'config_id') and
                        node.config_id == focal_node.config_id):
                    score *= 1.3

        return score

    def _truncate_value(self, value: Any, max_length: int = MAX_FIELD_LENGTH) -> str:
        """
        Truncate a value to a maximum length for serialization.

        Args:
            value: The value to truncate
            max_length: Maximum length to truncate to

        Returns:
            Truncated string representation of the value
        """
        str_value = str(value)
        if len(str_value) <= max_length:
            return str_value

        # Truncate with ellipsis
        return str_value[:max_length - 3] + "..."

    def _node_to_kgml(self, node: KGNode) -> str:
        """
        Convert a node to KGML format.

        Args:
            node: The KGNode to convert

        Returns:
            KGML string representation of the node
        """
        # Start with node identifier
        kgml = f"KGNODE► {node.uid} : "

        # Get node properties
        props = {}

        # Add type
        if hasattr(node, 'type'):
            props['type'] = node.type

        # Add common properties based on node type
        if hasattr(node, 'model_dump'):
            node_data = node.model_dump()

            # Filter out internal and redundant properties
            exclude_props = {'uid', 'meta_props', 'created_at', 'updated_at'}

            for key, value in node_data.items():
                if key not in exclude_props and value is not None:
                    props[key] = self._truncate_value(value)

        # Add meta properties if available
        if hasattr(node, 'meta_props') and node.meta_props:
            for key, value in node.meta_props.items():
                if value is not None:
                    props[f"meta.{key}"] = self._truncate_value(value)

        # Add timestamps
        if hasattr(node, 'created_at') and node.created_at:
            props['created_at'] = node.created_at
        if hasattr(node, 'updated_at') and node.updated_at:
            props['updated_at'] = node.updated_at

        # Serialize properties
        props_str = ", ".join([f'{k}="{v}"' for k, v in props.items()])
        kgml += props_str + " ◄\n"

        return kgml

    def _link_to_kgml(self, link: KGEdge) -> str:
        """
        Convert a link to KGML format.

        Args:
            link: The KGEdge to convert

        Returns:
            KGML string representation of the link
        """
        # Format: KGLINK► source_uid -> target_uid : relation="value", meta.prop="value" ◄
        kgml = f"KGLINK► {link.source_uid} -> {link.target_uid} : "

        # Get link properties
        props = {}

        # Add relation
        if hasattr(link, 'relation'):
            props['relation'] = link.relation

        # Add metadata if available
        if hasattr(link, 'meta_props') and link.meta_props:
            for key, value in link.meta_props.items():
                if value is not None:
                    props[f"meta.{key}"] = self._truncate_value(value)

        # Serialize properties
        props_str = ", ".join([f'{k}="{v}"' for k, v in props.items()])
        kgml += props_str + " ◄\n"

        return kgml

    # Replace the regex-based parsing with proper parser usage
    def parse_kgml_response(self, kgml_text: str) -> List[Dict[str, Any]]:
        """
        Parse a KGML response using the proper KGML parser.

        This method is maintained for backward compatibility but uses
        the proper parser infrastructure.

        Args:
            kgml_text: The KGML text to parse

        Returns:
            List of operation dictionaries
        """
        try:
            # Use the proper KGML parser
            tokens = tokenize(kgml_text)
            parser = Parser(tokens)
            ast = parser.parse_program()

            # Convert AST to the expected operation dictionaries format
            operations = self._ast_to_operations(ast)

            return operations
        except Exception as e:
            self.logger.error(f"Error parsing KGML: {str(e)}")
            return []

    def _ast_to_operations(self, ast) -> List[Dict[str, Any]]:
        """
        Convert a parsed AST to the operation dictionaries format.

        This is for backward compatibility with existing code that expects
        operations in dictionary format.

        Args:
            ast: The parsed AST

        Returns:
            List of operation dictionaries
        """
        operations = []

        # Process each statement in the program
        for statement in ast.statements:
            # Handle simple commands (C►, U►, D►, E►)
            if hasattr(statement, 'cmd_type') and statement.cmd_type in ['C►', 'U►', 'D►', 'E►']:
                operations.append({
                    'command': statement.cmd_type.rstrip('►'),  # Remove the trailing marker
                    'entity_type': statement.entity_type,
                    'entity_id': statement.uid,
                    'instruction': statement.instruction
                })
            # Handle conditional statements
            elif hasattr(statement, 'if_clause'):
                # Extract condition command
                condition_cmd, if_block = statement.if_clause

                operations.append({
                    'command': 'IF',
                    'condition': condition_cmd.instruction if hasattr(condition_cmd, 'instruction') else "",
                    'body': self._format_block(if_block)
                })

                # Handle elif clauses
                if hasattr(statement, 'elif_clauses') and statement.elif_clauses:
                    for i, (elif_cmd, elif_block) in enumerate(statement.elif_clauses):
                        operations.append({
                            'command': 'ELIF',
                            'condition': elif_cmd.instruction if hasattr(elif_cmd, 'instruction') else "",
                            'body': self._format_block(elif_block)
                        })

                # Handle else clause
                if hasattr(statement, 'else_clause') and statement.else_clause:
                    operations.append({
                        'command': 'ELSE',
                        'body': self._format_block(statement.else_clause)
                    })
            # Handle loop statements
            elif hasattr(statement, 'condition') and hasattr(statement, 'block'):
                operations.append({
                    'command': 'LOOP',
                    'instruction': statement.condition,
                    'body': self._format_block(statement.block)
                })
            # Handle KG blocks (could add if needed)

        return operations

    def _format_block(self, block) -> str:
        """
        Format a block of AST statements as a string.

        Args:
            block: List of AST statement nodes

        Returns:
            Formatted string representation of the block
        """
        result = ""
        for stmt in block:
            if hasattr(stmt, 'cmd_type') and stmt.cmd_type in ['C►', 'U►', 'D►', 'E►']:
                result += f"{stmt.cmd_type} {stmt.entity_type} {stmt.uid} \"{stmt.instruction}\" ◄\n"
            # Handle nested blocks if needed

        return result
