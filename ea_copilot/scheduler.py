"""
Scheduler for Executive Assistant
Runs prompts on schedule using cron-like syntax
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional
import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ms_graph import (
    MSGraphClient,
    format_calendar_events,
    format_emails,
    format_tasks
)
from llm_client import LLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EAScheduler:
    """Scheduler for Executive Assistant tasks."""

    def __init__(self, config_path: str = "config.yaml", prompts_path: str = "prompts.yaml"):
        self.config = self._load_yaml(config_path)
        self.prompts = self._load_yaml(prompts_path)
        self.scheduler = BlockingScheduler()
        self.graph_client = MSGraphClient()
        self.llm_client = LLMClient()
        self.user_email = self.config.get("user_email")

    def _load_yaml(self, path: str) -> dict:
        """Load YAML configuration file."""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _get_prompt_template(self, prompt_name: str) -> dict:
        """Get a prompt template by name."""
        return self.prompts.get("prompts", {}).get(prompt_name, {})

    def _gather_context(self, prompt_name: str) -> dict:
        """Gather Microsoft 365 data for prompt context."""
        context = {
            "date": datetime.now().strftime("%A, %B %d, %Y"),
            "time": datetime.now().strftime("%H:%M"),
            "week_start": (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%B %d, %Y")
        }

        try:
            # Calendar events for today
            today_events = self.graph_client.get_calendar_events()
            context["calendar_events"] = format_calendar_events(today_events)

            # Tomorrow's calendar
            tomorrow = datetime.now() + timedelta(days=1)
            tomorrow_end = tomorrow + timedelta(days=1)
            tomorrow_events = self.graph_client.get_calendar_events(tomorrow, tomorrow_end)
            context["tomorrow_calendar"] = format_calendar_events(tomorrow_events)

            # Next week's calendar
            next_week_start = datetime.now() + timedelta(days=7 - datetime.now().weekday())
            next_week_end = next_week_start + timedelta(days=5)
            next_week_events = self.graph_client.get_calendar_events(next_week_start, next_week_end)
            context["next_week_calendar"] = format_calendar_events(next_week_events)

            # Emails
            unread_emails = self.graph_client.get_emails(unread_only=True)
            context["recent_emails"] = format_emails(unread_emails)
            context["email_count"] = self.graph_client.get_email_count()

            # Tasks
            tasks = self.graph_client.get_tasks()
            context["tasks"] = format_tasks(tasks)
            context["pending_tasks"] = format_tasks([t for t in tasks if t.get("status") != "completed"])

            # Week statistics (simplified)
            week_start = datetime.now() - timedelta(days=datetime.now().weekday())
            week_events = self.graph_client.get_calendar_events(week_start, datetime.now())
            context["week_meetings"] = format_calendar_events(week_events)
            context["completed_meetings"] = format_calendar_events([e for e in today_events if datetime.fromisoformat(e.get("end", {}).get("dateTime", "").replace("Z", "+00:00")) < datetime.now()])

        except Exception as e:
            logger.error(f"Error gathering context: {e}")
            # Provide fallback empty values
            for key in ["calendar_events", "tomorrow_calendar", "recent_emails", "tasks"]:
                if key not in context:
                    context[key] = "Unable to retrieve data"

        return context

    def _render_prompt(self, template: str, context: dict) -> str:
        """Render prompt template with context."""
        result = template
        for key, value in context.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def _execute_task(self, task: dict):
        """Execute a scheduled task."""
        task_name = task.get("name", "Unknown")
        prompt_name = task.get("prompt")
        actions = task.get("actions", [])

        logger.info(f"Executing task: {task_name}")

        try:
            # Get prompt template
            prompt_template = self._get_prompt_template(prompt_name)
            if not prompt_template:
                logger.error(f"Prompt not found: {prompt_name}")
                return

            # Gather context from Microsoft 365
            context = self._gather_context(prompt_name)

            # Render the prompt
            system_prompt = prompt_template.get("system", "")
            user_prompt = self._render_prompt(prompt_template.get("template", ""), context)

            # Execute with LLM
            response = self.llm_client.complete(
                system=system_prompt,
                prompt=user_prompt
            )

            # Check for NO_ALERT response (for priority email checks)
            if "NO_ALERT" in response:
                logger.info(f"Task {task_name}: No alert needed")
                return

            # Execute actions
            for action in actions:
                if action == "send_email":
                    self._send_email_action(task_name, prompt_template.get("name", task_name), response)
                elif action == "teams_notify":
                    self._teams_notify_action(task_name, response)
                elif action == "log":
                    logger.info(f"Task {task_name} result:\n{response}")

            logger.info(f"Task {task_name} completed successfully")

        except Exception as e:
            logger.error(f"Error executing task {task_name}: {e}")

    def _send_email_action(self, task_name: str, subject_prefix: str, content: str):
        """Send email with task results."""
        subject = f"[EA] {subject_prefix} - {datetime.now().strftime('%Y-%m-%d')}"

        # Convert to HTML
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2>{subject_prefix}</h2>
        <div style="white-space: pre-wrap;">{content}</div>
        <hr>
        <p style="color: #666; font-size: 12px;">
            Generated by Executive Assistant at {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </p>
        </body>
        </html>
        """

        self.graph_client.send_email(
            to=self.user_email,
            subject=subject,
            body=html_content,
            body_type="HTML"
        )
        logger.info(f"Email sent for task: {task_name}")

    def _teams_notify_action(self, task_name: str, content: str):
        """Send Teams notification (webhook)."""
        webhook_url = self.config.get("notifications", {}).get("teams", {}).get("webhook_url")
        if not webhook_url:
            logger.warning("Teams webhook URL not configured")
            return

        import requests
        payload = {
            "text": f"**{task_name}**\n\n{content}"
        }
        requests.post(webhook_url, json=payload)
        logger.info(f"Teams notification sent for task: {task_name}")

    def _setup_meeting_prep_task(self, task: dict):
        """Set up meeting preparation task (runs before each meeting)."""
        minutes_before = task.get("minutes_before", 15)

        def check_and_prep_meetings():
            """Check for upcoming meetings and prepare."""
            try:
                upcoming = self.graph_client.get_upcoming_meetings(minutes=minutes_before + 5)
                for meeting in upcoming:
                    start_time = datetime.fromisoformat(
                        meeting.get("start", {}).get("dateTime", "").replace("Z", "+00:00")
                    )
                    time_until = (start_time - datetime.now()).total_seconds() / 60

                    # If meeting is within our window
                    if 0 < time_until <= minutes_before:
                        # Prepare context for this specific meeting
                        context = {
                            "meeting_title": meeting.get("subject", "Untitled Meeting"),
                            "meeting_time": start_time.strftime("%H:%M"),
                            "attendees": ", ".join([
                                a.get("emailAddress", {}).get("name", "")
                                for a in meeting.get("attendees", [])
                            ]),
                            "related_emails": "",
                            "previous_meetings": ""
                        }

                        # Search for related emails
                        for attendee in meeting.get("attendees", [])[:3]:
                            email_addr = attendee.get("emailAddress", {}).get("address", "")
                            if email_addr:
                                related = self.graph_client.search_emails(email_addr, top=3)
                                context["related_emails"] += format_emails(related)

                        prompt_template = self._get_prompt_template("meeting_preparation")
                        system_prompt = prompt_template.get("system", "")
                        user_prompt = self._render_prompt(prompt_template.get("template", ""), context)

                        response = self.llm_client.complete(system=system_prompt, prompt=user_prompt)
                        self._send_email_action(
                            task.get("name"),
                            f"Meeting Prep: {meeting.get('subject', 'Meeting')}",
                            response
                        )

            except Exception as e:
                logger.error(f"Error in meeting prep task: {e}")

        # Run every 5 minutes to check for upcoming meetings
        self.scheduler.add_job(
            check_and_prep_meetings,
            CronTrigger.from_crontab("*/5 * * * *"),
            id=f"meeting_prep_{task.get('name')}"
        )

    def setup_tasks(self):
        """Set up all scheduled tasks from config."""
        tasks = self.config.get("tasks", [])

        for task in tasks:
            if not task.get("enabled", True):
                logger.info(f"Task {task.get('name')} is disabled, skipping")
                continue

            schedule = task.get("schedule")

            if schedule == "before_meeting":
                self._setup_meeting_prep_task(task)
            else:
                # Regular cron-based task
                try:
                    trigger = CronTrigger.from_crontab(schedule)
                    self.scheduler.add_job(
                        self._execute_task,
                        trigger,
                        args=[task],
                        id=task.get("name")
                    )
                    logger.info(f"Scheduled task: {task.get('name')} with schedule: {schedule}")
                except Exception as e:
                    logger.error(f"Error scheduling task {task.get('name')}: {e}")

    def run(self):
        """Start the scheduler."""
        logger.info("Starting Executive Assistant Scheduler...")
        self.setup_tasks()

        logger.info("Scheduled tasks:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  - {job.id}: next run at {job.next_run_time}")

        try:
            self.scheduler.start()
        except KeyboardInterrupt:
            logger.info("Shutting down scheduler...")
            self.scheduler.shutdown()

    def run_once(self, task_name: str):
        """Run a specific task once (for testing)."""
        tasks = self.config.get("tasks", [])
        for task in tasks:
            if task.get("name") == task_name:
                self._execute_task(task)
                return
        logger.error(f"Task not found: {task_name}")

    def run_all_once(self):
        """Run all enabled tasks once (for testing)."""
        tasks = self.config.get("tasks", [])
        for task in tasks:
            if task.get("enabled", True):
                self._execute_task(task)
