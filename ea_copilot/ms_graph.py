"""
Microsoft Graph API Integration for Executive Assistant
Handles authentication and data retrieval from Microsoft 365
"""

import os
from datetime import datetime, timedelta
from typing import Optional
import msal
import requests


class MSGraphClient:
    """Client for Microsoft Graph API interactions."""

    GRAPH_URL = "https://graph.microsoft.com/v1.0"
    SCOPES = [
        "https://graph.microsoft.com/.default"
    ]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None
    ):
        self.client_id = client_id or os.getenv("MICROSOFT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("MICROSOFT_CLIENT_SECRET")
        self.tenant_id = tenant_id or os.getenv("MICROSOFT_TENANT_ID")
        self.user_email = os.getenv("MICROSOFT_USER_EMAIL")
        self._token = None
        self._token_expires = None

    def _get_token(self) -> str:
        """Get or refresh access token."""
        if self._token and self._token_expires and datetime.now() < self._token_expires:
            return self._token

        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
            client_credential=self.client_secret
        )

        result = app.acquire_token_for_client(scopes=self.SCOPES)

        if "access_token" in result:
            self._token = result["access_token"]
            self._token_expires = datetime.now() + timedelta(seconds=result.get("expires_in", 3600) - 60)
            return self._token
        else:
            raise Exception(f"Failed to get token: {result.get('error_description', 'Unknown error')}")

    def _headers(self) -> dict:
        """Get request headers with authorization."""
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json"
        }

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request to Graph API."""
        url = f"{self.GRAPH_URL}{endpoint}"
        response = requests.get(url, headers=self._headers(), params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: dict) -> dict:
        """Make POST request to Graph API."""
        url = f"{self.GRAPH_URL}{endpoint}"
        response = requests.post(url, headers=self._headers(), json=data)
        response.raise_for_status()
        return response.json()

    # Calendar Methods
    def get_calendar_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user: Optional[str] = None
    ) -> list:
        """Get calendar events for a date range."""
        user = user or self.user_email
        start = start_date or datetime.now()
        end = end_date or (start + timedelta(days=1))

        endpoint = f"/users/{user}/calendar/calendarView"
        params = {
            "startDateTime": start.isoformat() + "Z",
            "endDateTime": end.isoformat() + "Z",
            "$orderby": "start/dateTime",
            "$select": "subject,start,end,attendees,location,bodyPreview,organizer"
        }

        result = self._get(endpoint, params)
        return result.get("value", [])

    def get_upcoming_meetings(self, minutes: int = 60, user: Optional[str] = None) -> list:
        """Get meetings in the next N minutes."""
        start = datetime.now()
        end = start + timedelta(minutes=minutes)
        return self.get_calendar_events(start, end, user)

    # Email Methods
    def get_emails(
        self,
        folder: str = "inbox",
        unread_only: bool = False,
        top: int = 20,
        user: Optional[str] = None
    ) -> list:
        """Get emails from a folder."""
        user = user or self.user_email
        endpoint = f"/users/{user}/mailFolders/{folder}/messages"

        params = {
            "$top": top,
            "$orderby": "receivedDateTime desc",
            "$select": "subject,from,receivedDateTime,bodyPreview,importance,isRead"
        }

        if unread_only:
            params["$filter"] = "isRead eq false"

        result = self._get(endpoint, params)
        return result.get("value", [])

    def get_email_count(self, folder: str = "inbox", unread_only: bool = True, user: Optional[str] = None) -> int:
        """Get count of emails in folder."""
        user = user or self.user_email
        endpoint = f"/users/{user}/mailFolders/{folder}"
        result = self._get(endpoint)
        if unread_only:
            return result.get("unreadItemCount", 0)
        return result.get("totalItemCount", 0)

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        body_type: str = "HTML",
        user: Optional[str] = None
    ) -> dict:
        """Send an email."""
        user = user or self.user_email
        endpoint = f"/users/{user}/sendMail"

        message = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": body_type,
                    "content": body
                },
                "toRecipients": [
                    {"emailAddress": {"address": to}}
                ]
            }
        }

        return self._post(endpoint, message)

    def search_emails(
        self,
        query: str,
        top: int = 10,
        user: Optional[str] = None
    ) -> list:
        """Search emails by query."""
        user = user or self.user_email
        endpoint = f"/users/{user}/messages"

        params = {
            "$search": f'"{query}"',
            "$top": top,
            "$select": "subject,from,receivedDateTime,bodyPreview"
        }

        result = self._get(endpoint, params)
        return result.get("value", [])

    # Tasks Methods (Microsoft To Do)
    def get_tasks(
        self,
        list_name: str = "Tasks",
        completed: bool = False,
        user: Optional[str] = None
    ) -> list:
        """Get tasks from To Do."""
        user = user or self.user_email

        # First get task lists
        lists_endpoint = f"/users/{user}/todo/lists"
        lists_result = self._get(lists_endpoint)

        task_list = None
        for tl in lists_result.get("value", []):
            if tl.get("displayName", "").lower() == list_name.lower():
                task_list = tl
                break

        if not task_list:
            return []

        # Get tasks from the list
        list_id = task_list["id"]
        tasks_endpoint = f"/users/{user}/todo/lists/{list_id}/tasks"

        params = {"$orderby": "importance desc,dueDateTime/dateTime"}
        if not completed:
            params["$filter"] = "status ne 'completed'"

        result = self._get(tasks_endpoint, params)
        return result.get("value", [])

    # People/Contacts Methods
    def get_relevant_people(self, user: Optional[str] = None) -> list:
        """Get people relevant to the user."""
        user = user or self.user_email
        endpoint = f"/users/{user}/people"
        result = self._get(endpoint)
        return result.get("value", [])

    def search_people(self, query: str, user: Optional[str] = None) -> list:
        """Search for people."""
        user = user or self.user_email
        endpoint = f"/users/{user}/people"
        params = {"$search": query}
        result = self._get(endpoint, params)
        return result.get("value", [])


def format_calendar_events(events: list) -> str:
    """Format calendar events for prompt."""
    if not events:
        return "No events scheduled."

    lines = []
    for event in events:
        start = event.get("start", {}).get("dateTime", "")
        end = event.get("end", {}).get("dateTime", "")
        subject = event.get("subject", "No subject")
        attendees = [a.get("emailAddress", {}).get("name", "") for a in event.get("attendees", [])]

        if start:
            start_time = datetime.fromisoformat(start.replace("Z", "+00:00")).strftime("%H:%M")
            end_time = datetime.fromisoformat(end.replace("Z", "+00:00")).strftime("%H:%M")
            time_str = f"{start_time}-{end_time}"
        else:
            time_str = "All day"

        attendee_str = ", ".join(attendees[:3])
        if len(attendees) > 3:
            attendee_str += f" +{len(attendees) - 3} others"

        lines.append(f"- {time_str}: {subject}")
        if attendees:
            lines.append(f"  Attendees: {attendee_str}")

    return "\n".join(lines)


def format_emails(emails: list) -> str:
    """Format emails for prompt."""
    if not emails:
        return "No emails."

    lines = []
    for email in emails:
        sender = email.get("from", {}).get("emailAddress", {}).get("name", "Unknown")
        subject = email.get("subject", "No subject")
        preview = email.get("bodyPreview", "")[:100]
        importance = email.get("importance", "normal")

        priority_marker = "🔴 " if importance == "high" else ""
        lines.append(f"- {priority_marker}From: {sender}")
        lines.append(f"  Subject: {subject}")
        lines.append(f"  Preview: {preview}...")
        lines.append("")

    return "\n".join(lines)


def format_tasks(tasks: list) -> str:
    """Format tasks for prompt."""
    if not tasks:
        return "No pending tasks."

    lines = []
    for task in tasks:
        title = task.get("title", "Untitled")
        importance = task.get("importance", "normal")
        due = task.get("dueDateTime", {}).get("dateTime", "")

        priority_marker = "🔴 " if importance == "high" else ""
        due_str = ""
        if due:
            due_date = datetime.fromisoformat(due.replace("Z", "+00:00")).strftime("%Y-%m-%d")
            due_str = f" (Due: {due_date})"

        lines.append(f"- {priority_marker}{title}{due_str}")

    return "\n".join(lines)
