from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatInfo(BaseModel):
    """Model for chat information."""
    chat_id: str
    user_id: str


class ActionContext(BaseModel):
    """Model for storing context information between actions."""
    # Entity (user or organization) information
    entity_name: Optional[str] = None  # User or org name
    entity_type: Optional[str] = None  # "User" or "Organization"
    entity_id: Optional[str] = None  # User or org node ID

    # Project information
    project_number: Optional[int] = None
    project_id: Optional[str] = None
    project_title: Optional[str] = None

    # Repository information
    repo_name: Optional[str] = None
    repo_owner: Optional[str] = None
    repo_id: Optional[str] = None

    # Item information
    item_id: Optional[str] = None
    item_title: Optional[str] = None

    # Field information
    field_id: Optional[str] = None
    field_name: Optional[str] = None
    field_type: Optional[str] = None
    field_options: Optional[Dict[str, str]] = None

    # Option information
    selected_option_id: Optional[str] = None
    selected_option_name: Optional[str] = None

    # Iteration information
    iterations: Optional[Dict[str, str]] = None
    selected_iteration_id: Optional[str] = None

    # Project custom fields
    custom_fields: Optional[List[Dict[str, Any]]] = None

    # Common field types in GitHub projects
    issue_types: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    labels: Optional[List[str]] = None

    def get_custom_field_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a custom field by name (case-insensitive)."""
        if not self.custom_fields:
            return None

        for field in self.custom_fields:
            if field.get("name", "").lower() == name.lower():
                return field

        return None

    def get_field_options(self, field_name: str) -> List[str]:
        """Get options for a select field by name."""
        field = self.get_custom_field_by_name(field_name)
        if not field:
            return []

        if field.get("type") != "single_select":
            return []

        options = []
        if "options" in field:
            options = list(field.get("options", {}).values())
        elif "configuration" in field:
            config_options = field.get("configuration", {}).get("options", [])
            options = [opt.get("name") for opt in config_options if "name" in opt]

        return options

    def get_option_id(self, field_name: str, option_name: str) -> Optional[str]:
        """Get option ID for a given field and option name."""
        field = self.get_custom_field_by_name(field_name)
        if not field:
            return None

        if field.get("type") != "single_select":
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

    def add_custom_field_info(self, field: Dict[str, Any]) -> None:
        """Add or update a custom field in the context."""
        if not field or "name" not in field:
            return

        if not self.custom_fields:
            self.custom_fields = []

        # Check if field already exists
        existing_idx = -1
        for i, existing in enumerate(self.custom_fields):
            if existing.get("name") == field.get("name"):
                existing_idx = i
                break

        if existing_idx >= 0:
            # Update existing field
            self.custom_fields[existing_idx] = field
        else:
            # Add new field
            self.custom_fields.append(field)

        # Update specialized field lists
        self._update_specialized_fields()

    def _update_specialized_fields(self) -> None:
        """Update specialized field lists based on custom fields."""
        if not self.custom_fields:
            return

        # Reset lists
        self.issue_types = []
        self.statuses = []
        self.categories = []
        self.labels = []

        for field in self.custom_fields:
            field_name = field.get("name", "").lower()

            if field.get("type") != "single_select":
                continue

            options = []
            if "options" in field:
                options = list(field.get("options", {}).values())
            elif "configuration" in field:
                config_options = field.get("configuration", {}).get("options", [])
                options = [opt.get("name") for opt in config_options if "name" in opt]

            # Process based on field name
            if field_name in ["type", "issue type", "item type"]:
                self.issue_types = options
            elif field_name in ["status", "state"]:
                self.statuses = options
            elif field_name in ["category", "area", "component"]:
                self.categories = options
            elif field_name in ["label", "labels", "tag", "tags"]:
                self.labels = options