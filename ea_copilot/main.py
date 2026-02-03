#!/usr/bin/env python3
"""
Executive Assistant for Microsoft 365 Copilot
Automated scheduling of AI-powered executive assistant tasks

Usage:
    python main.py                  # Start scheduler (runs continuously)
    python main.py run              # Start scheduler
    python main.py test             # Run all tasks once for testing
    python main.py test <task>      # Run specific task once
    python main.py list             # List configured tasks
    python main.py setup            # Interactive setup wizard
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scheduler import EAScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def check_env_vars() -> bool:
    """Check if required environment variables are set."""
    required_vars = {
        "Microsoft 365": [
            "MICROSOFT_CLIENT_ID",
            "MICROSOFT_CLIENT_SECRET",
            "MICROSOFT_TENANT_ID",
            "MICROSOFT_USER_EMAIL"
        ],
        "LLM (Azure OpenAI)": [
            "AZURE_OPENAI_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT"
        ]
    }

    all_set = True
    for category, vars in required_vars.items():
        missing = [v for v in vars if not os.getenv(v)]
        if missing:
            logger.warning(f"Missing {category} environment variables: {', '.join(missing)}")
            all_set = False

    return all_set


def print_setup_instructions():
    """Print setup instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Executive Assistant Setup Instructions                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. MICROSOFT 365 SETUP (Azure AD App Registration)
   ─────────────────────────────────────────────────
   a. Go to Azure Portal → Azure Active Directory → App registrations
   b. Create new registration:
      - Name: "Executive Assistant"
      - Supported account types: Single tenant
   c. Add API permissions:
      - Microsoft Graph → Application permissions:
        • Calendars.Read
        • Mail.Read
        • Mail.Send
        • Tasks.Read
        • People.Read
        • User.Read.All
   d. Grant admin consent for permissions
   e. Create client secret (Certificates & secrets)
   f. Note down: Client ID, Tenant ID, Client Secret

2. AZURE OPENAI SETUP (or use OpenAI)
   ────────────────────────────────────
   a. Create Azure OpenAI resource in Azure Portal
   b. Deploy a model (e.g., gpt-4)
   c. Note down: Endpoint, API Key, Deployment name

3. ENVIRONMENT VARIABLES
   ──────────────────────
   Create a .env file or export these variables:

   # Microsoft 365
   export MICROSOFT_CLIENT_ID="your-client-id"
   export MICROSOFT_CLIENT_SECRET="your-client-secret"
   export MICROSOFT_TENANT_ID="your-tenant-id"
   export MICROSOFT_USER_EMAIL="your-email@company.com"

   # Azure OpenAI
   export AZURE_OPENAI_KEY="your-api-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
   export AZURE_OPENAI_DEPLOYMENT="gpt-4"

   # OR use OpenAI instead
   export LLM_PROVIDER="openai"
   export OPENAI_API_KEY="your-openai-key"

4. CONFIGURE TASKS
   ────────────────
   Edit config.yaml to customize:
   - Your email address
   - Task schedules (cron format)
   - Enable/disable tasks

5. RUN
   ───
   python main.py run        # Start scheduler
   python main.py test       # Test all tasks once

""")


def list_tasks(scheduler: EAScheduler):
    """List all configured tasks."""
    print("\n📋 Configured Tasks")
    print("=" * 60)

    tasks = scheduler.config.get("tasks", [])
    for task in tasks:
        status = "✅ Enabled" if task.get("enabled", True) else "❌ Disabled"
        print(f"\n{task.get('name')}")
        print(f"  Status:   {status}")
        print(f"  Schedule: {task.get('schedule')}")
        print(f"  Prompt:   {task.get('prompt')}")
        print(f"  Actions:  {', '.join(task.get('actions', []))}")

    print("\n")


def run_interactive_setup():
    """Run interactive setup wizard."""
    print("\n🔧 Executive Assistant Setup Wizard\n")

    # Check for existing .env
    env_path = Path(__file__).parent / ".env"

    config = {}

    print("Microsoft 365 Configuration")
    print("-" * 40)
    config["MICROSOFT_CLIENT_ID"] = input("Client ID: ").strip()
    config["MICROSOFT_CLIENT_SECRET"] = input("Client Secret: ").strip()
    config["MICROSOFT_TENANT_ID"] = input("Tenant ID: ").strip()
    config["MICROSOFT_USER_EMAIL"] = input("Your Email: ").strip()

    print("\nLLM Configuration")
    print("-" * 40)
    provider = input("Provider (azure/openai) [azure]: ").strip() or "azure"
    config["LLM_PROVIDER"] = provider

    if provider == "azure":
        config["AZURE_OPENAI_KEY"] = input("Azure OpenAI Key: ").strip()
        config["AZURE_OPENAI_ENDPOINT"] = input("Azure OpenAI Endpoint: ").strip()
        config["AZURE_OPENAI_DEPLOYMENT"] = input("Deployment Name [gpt-4]: ").strip() or "gpt-4"
    else:
        config["OPENAI_API_KEY"] = input("OpenAI API Key: ").strip()

    # Write .env file
    with open(env_path, "w") as f:
        for key, value in config.items():
            f.write(f'{key}="{value}"\n')

    print(f"\n✅ Configuration saved to {env_path}")
    print("   Run 'source .env' to load environment variables")
    print("   Then run 'python main.py test' to verify setup\n")


def main():
    parser = argparse.ArgumentParser(
        description="Executive Assistant for Microsoft 365 Copilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run              Start the scheduler
  python main.py test             Test all tasks once
  python main.py test morning_briefing    Test specific task
  python main.py list             Show configured tasks
  python main.py setup            Run setup wizard
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "test", "list", "setup", "help"],
        help="Command to execute"
    )

    parser.add_argument(
        "task_name",
        nargs="?",
        help="Task name for 'test' command"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--prompts",
        default="prompts.yaml",
        help="Path to prompts file"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM client for testing"
    )

    args = parser.parse_args()

    # Change to script directory
    os.chdir(Path(__file__).parent)

    if args.command == "help":
        print_setup_instructions()
        return

    if args.command == "setup":
        run_interactive_setup()
        return

    # Check environment
    if not check_env_vars():
        print("\n⚠️  Missing environment variables!")
        print("   Run 'python main.py help' for setup instructions")
        print("   Or run 'python main.py setup' for interactive setup\n")

        if args.command != "list":
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                return

    # Initialize scheduler
    try:
        scheduler = EAScheduler(
            config_path=args.config,
            prompts_path=args.prompts
        )
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return

    if args.mock:
        from llm_client import MockLLMClient
        scheduler.llm_client = MockLLMClient()
        logger.info("Using mock LLM client")

    if args.command == "list":
        list_tasks(scheduler)

    elif args.command == "test":
        if args.task_name:
            logger.info(f"Running task: {args.task_name}")
            scheduler.run_once(args.task_name)
        else:
            logger.info("Running all tasks once...")
            scheduler.run_all_once()

    elif args.command == "run":
        logger.info("Starting Executive Assistant...")
        scheduler.run()


if __name__ == "__main__":
    main()
