#!/bin/bash
# Script to create the paddleaurum project structure
set -e  # Exit on any error

echo "Creating paddleaurum project structure..."

# Create root directory
mkdir -p paddleaurum

# Create top-level files
touch paddleaurum/README.md
touch paddleaurum/requirements.txt
touch paddleaurum/.env.example
touch paddleaurum/main.py

# Create config directory and files
mkdir -p paddleaurum/config
touch paddleaurum/config/agents.yaml
touch paddleaurum/config/tasks.yaml
touch paddleaurum/config/settings.py

# Create agents directory and files
mkdir -p paddleaurum/agents
touch paddleaurum/agents/__init__.py
touch paddleaurum/agents/ceo.py
touch paddleaurum/agents/customer_captain.py
touch paddleaurum/agents/chat_buddy.py
touch paddleaurum/agents/stock_sergeant.py
touch paddleaurum/agents/stock_scout.py
touch paddleaurum/agents/promo_general.py
touch paddleaurum/agents/recommender.py

# Create tasks directory and files
mkdir -p paddleaurum/tasks
touch paddleaurum/tasks/__init__.py
touch paddleaurum/tasks/support_tasks.py
touch paddleaurum/tasks/inventory_tasks.py
touch paddleaurum/tasks/marketing_tasks.py
touch paddleaurum/tasks/recommendation_tasks.py

# Create tools directory and files
mkdir -p paddleaurum/tools
touch paddleaurum/tools/__init__.py
touch paddleaurum/tools/shopify_tool.py
touch paddleaurum/tools/search_tool.py
touch paddleaurum/tools/browser_tool.py
touch paddleaurum/tools/db_tool.py
touch paddleaurum/tools/email_tool.py

# Create memory directory and files
mkdir -p paddleaurum/memory
touch paddleaurum/memory/__init__.py
touch paddleaurum/memory/agent_memory.db
touch paddleaurum/memory/memory_manager.py

# Create models directory and files
mkdir -p paddleaurum/models
touch paddleaurum/models/__init__.py
touch paddleaurum/models/ollama_loader.py
touch paddleaurum/models/model_config.py

# Create diagrams directory and files
mkdir -p paddleaurum/diagrams
touch paddleaurum/diagrams/chat_buddy_flowchart.mmd
touch paddleaurum/diagrams/chat_buddy_sequence.mmd
touch paddleaurum/diagrams/chat_buddy_class.mmd
touch paddleaurum/diagrams/chat_buddy_state.mmd
touch paddleaurum/diagrams/paddleaurum_chat_buddy_diagrams.html

# Create workflows directory and files
mkdir -p paddleaurum/workflows
touch paddleaurum/workflows/__init__.py
touch paddleaurum/workflows/launch_product.py
touch paddleaurum/workflows/restock_alert.py
touch paddleaurum/workflows/promo_campaign.py
touch paddleaurum/workflows/customer_support.py

# Create crews directory and files
mkdir -p paddleaurum/crews
touch paddleaurum/crews/__init__.py
touch paddleaurum/crews/main_crew.py
touch paddleaurum/crews/support_crew.py
touch paddleaurum/crews/inventory_crew.py
touch paddleaurum/crews/marketing_crew.py

# Create tests directory and subdirectories/files
mkdir -p paddleaurum/tests/fixtures
touch paddleaurum/tests/__init__.py
touch paddleaurum/tests/test_agents.py
touch paddleaurum/tests/test_tools.py
touch paddleaurum/tests/test_workflows.py
touch paddleaurum/tests/fixtures/mock_shopify_response.json
touch paddleaurum/tests/fixtures/mock_chat_input.json

# Create logs directory and files
mkdir -p paddleaurum/logs
touch paddleaurum/logs/agent_runs.log
touch paddleaurum/logs/errors.log

echo "Done! Project structure created under ./paddleaurum"