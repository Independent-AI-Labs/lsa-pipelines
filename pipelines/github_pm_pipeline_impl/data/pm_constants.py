# github_pm/config/env.py
import logging
import os

# GitHub API configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

BASE_API_URL = os.getenv("GITHUB_API_URL", "https://api.github.com")

# Model configuration
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "qwen2.5-coder:14b")
MANAGER_MODEL = os.getenv("MANAGER_MODEL", "qwen2.5-coder:14b")

# Define default owners for projects
GITHUB_USER = os.getenv("GITHUB_USER", "")
GITHUB_ORG = os.getenv("GITHUB_ORG", "")

# Define repository for testing
TEST_REPOSITORY = os.getenv("GITHUB_TEST_REPO", "test-repository")
TEST_PROJECT_NAME = os.getenv("GITHUB_TEST_PROJECT_NAME", "Test Project")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Rate limiting
API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "5000"))  # Default GitHub API rate limit

# Pipeline execution parameters
MAX_ACTIONS = int(os.getenv("MAX_ACTIONS", "10"))  # Maximum number of actions in a chain
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "20"))  # Default limit for pagination requests

# Timeout configuration
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # Timeout for API requests in seconds
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))  # Timeout for LLM requests in seconds

# Feature flags
ENABLE_RATE_LIMIT_HANDLING = os.getenv("ENABLE_RATE_LIMIT_HANDLING", "true").lower() == "true"
ENABLE_PARAMETER_VALIDATION = os.getenv("ENABLE_PARAMETER_VALIDATION", "true").lower() == "true"
ENABLE_PROJECT_DISCOVERY = os.getenv("ENABLE_PROJECT_DISCOVERY", "true").lower() == "true"
ENABLE_DRAFT_CONVERSION = os.getenv("ENABLE_DRAFT_CONVERSION", "true").lower() == "true"

# Default label settings
DEFAULT_LABELS = [label.strip() for label in os.getenv("GITHUB_DEFAULT_LABELS", "").split(",") if label.strip()]

# Cache settings
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # Cache TTL in seconds

# Debug settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
if DEBUG_MODE:
    LOG_LEVEL = logging.DEBUG
