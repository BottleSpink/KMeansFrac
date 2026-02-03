# Executive Assistant for Microsoft 365 Copilot

Automated AI-powered executive assistant that runs on schedule—not chat. Configure once, run continuously.

## Features

- **Scheduled Tasks**: Cron-based scheduling for automated execution
- **Morning Briefings**: Daily summary of calendar, emails, and tasks
- **Meeting Prep**: Automatic context and talking points before meetings
- **End-of-Day Summary**: Recap accomplishments and plan for tomorrow
- **Weekly Reports**: High-level executive summary
- **Priority Alerts**: Hourly check for urgent emails

## Quick Start

### 1. Install Dependencies

```bash
cd ea_copilot
pip install -r requirements.txt
```

### 2. Set Up Microsoft 365 Access

Create an Azure AD App Registration with these permissions:
- `Calendars.Read`
- `Mail.Read`
- `Mail.Send`
- `Tasks.Read`
- `People.Read`
- `User.Read.All`

### 3. Configure Environment

```bash
# Microsoft 365
export MICROSOFT_CLIENT_ID="your-client-id"
export MICROSOFT_CLIENT_SECRET="your-client-secret"
export MICROSOFT_TENANT_ID="your-tenant-id"
export MICROSOFT_USER_EMAIL="you@company.com"

# Azure OpenAI
export AZURE_OPENAI_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"
```

### 4. Customize Schedule

Edit `config.yaml` to set your email and task schedules:

```yaml
user_email: "you@company.com"

tasks:
  - name: "morning_briefing"
    schedule: "0 7 * * 1-5"  # 7 AM weekdays
    enabled: true
    prompt: "daily_briefing"
    actions:
      - send_email
```

### 5. Run

```bash
# Start scheduler (runs continuously)
python main.py run

# Test all tasks once
python main.py test

# Test specific task
python main.py test morning_briefing

# List configured tasks
python main.py list
```

## Configuration

### Schedule Format (Cron)

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday = 0)
│ │ │ │ │
│ │ │ │ │
* * * * *
```

Examples:
- `0 7 * * 1-5` - 7:00 AM, Monday-Friday
- `0 9-17 * * *` - Every hour 9 AM to 5 PM
- `0 16 * * 5` - 4:00 PM on Fridays

### Special Schedules

- `before_meeting` - Runs X minutes before each meeting

## Prompts

Edit `prompts.yaml` to customize AI prompts. Each prompt has:
- `system`: System instructions for the AI
- `template`: User prompt with `{{variables}}` for context

## LLM Providers

Supports multiple providers:

**Azure OpenAI (default)**
```bash
export LLM_PROVIDER="azure"
export AZURE_OPENAI_KEY="..."
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_DEPLOYMENT="gpt-4"
```

**OpenAI**
```bash
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="..."
```

## Running as a Service

### Linux (systemd)

Create `/etc/systemd/system/ea-copilot.service`:

```ini
[Unit]
Description=Executive Assistant
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/path/to/ea_copilot
EnvironmentFile=/path/to/ea_copilot/.env
ExecStart=/usr/bin/python3 main.py run
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ea-copilot
sudo systemctl start ea-copilot
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py", "run"]
```

## Architecture

```
ea_copilot/
├── main.py           # Entry point & CLI
├── scheduler.py      # Task scheduling engine
├── ms_graph.py       # Microsoft 365 API integration
├── llm_client.py     # AI/LLM integration
├── config.yaml       # Schedule configuration
├── prompts.yaml      # AI prompt templates
└── requirements.txt  # Dependencies
```

## License

MIT
