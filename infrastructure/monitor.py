import json
import boto3
import datetime
import subprocess
import os


def log_to_journal(message, priority="info"):
    """Log to systemd journal"""
    try:
        subprocess.run(
            ["systemd-cat", "-t", "zorkgpt-monitor", "-p", priority],
            input=message,
            text=True,
            check=True,
        )
    except:
        print(f"Failed to log to journal: {message}")


def check_health():
    timestamp = datetime.datetime.now().isoformat()

    # Check systemd service
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "zorkgpt"], capture_output=True, text=True
        )
        service_active = result.stdout.strip() == "active"
    except:
        service_active = False

    # Check if process is actually running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "main.py"], capture_output=True, text=True
        )
        process_running = bool(result.stdout.strip())
    except:
        process_running = False

    # Check current_state.json age (if it exists locally)
    state_file_fresh = True
    state_age_minutes = 0
    try:
        if os.path.exists("/home/zorkgpt/ZorkGPT/current_state.json"):
            with open("/home/zorkgpt/ZorkGPT/current_state.json", "r") as f:
                state_data = json.load(f)
                state_timestamp_str = state_data["metadata"]["timestamp"]

                # Handle timezone properly
                if state_timestamp_str.endswith("Z"):
                    state_timestamp = datetime.datetime.fromisoformat(
                        state_timestamp_str.replace("Z", "+00:00")
                    )
                else:
                    state_timestamp = datetime.datetime.fromisoformat(
                        state_timestamp_str
                    )
                    if state_timestamp.tzinfo is None:
                        state_timestamp = state_timestamp.replace(
                            tzinfo=datetime.timezone.utc
                        )

                now_utc = datetime.datetime.now(datetime.timezone.utc)
                state_age_minutes = (now_utc - state_timestamp).total_seconds() / 60
                state_file_fresh = state_age_minutes < 10  # Alert if > 10 minutes old
    except Exception as e:
        state_file_fresh = False
        log_to_journal(f"Error checking state file: {e}", "warning")

    health_data = {
        "timestamp": timestamp,
        "service_active": service_active,
        "process_running": process_running,
        "state_file_fresh": state_file_fresh,
        "state_age_minutes": round(state_age_minutes, 1),
    }

    # Log to journal (structured logging)
    log_message = json.dumps(health_data)

    # Determine log level based on health
    if service_active and process_running and state_file_fresh:
        log_to_journal(f"Health check OK: {log_message}", "info")
    else:
        log_to_journal(f"Health check FAILED: {log_message}", "err")
        send_alert(health_data)

    return health_data


def send_alert(health_data):
    try:
        topic_arn = os.environ.get("ALERT_TOPIC_ARN")
        if not topic_arn:
            log_to_journal("No ALERT_TOPIC_ARN set, skipping alert", "warning")
            return

        # Create SNS client with explicit region
        sns = boto3.client("sns", region_name="us-east-1")

        issues = []
        if not health_data["service_active"]:
            issues.append("❌ Systemd service not active")
        if not health_data["process_running"]:
            issues.append("❌ Python process not running")
        if not health_data["state_file_fresh"]:
            issues.append(
                f"❌ State file stale ({health_data['state_age_minutes']} min old)"
            )

        message = f"""🚨 ZorkGPT Health Alert

Time: {health_data["timestamp"]}

Issues Detected:
{chr(10).join(issues)}

Status Summary:
• Service Active: {health_data["service_active"]}
• Process Running: {health_data["process_running"]}
• State File Fresh: {health_data["state_file_fresh"]}
• State Age: {health_data["state_age_minutes"]} minutes

Check logs: journalctl -u zorkgpt -f
Check monitor: journalctl -t zorkgpt-monitor -f
"""

        sns.publish(TopicArn=topic_arn, Subject="🚨 ZorkGPT Alert", Message=message)

        log_to_journal(f"Alert sent: {', '.join(issues)}", "notice")

    except Exception as e:
        log_to_journal(f"Failed to send alert: {e}", "err")


if __name__ == "__main__":
    check_health()
