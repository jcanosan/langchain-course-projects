# Tools which reflect different actions in the simulated email account

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

MOCK_USERS = {
    "user@example.com": "password123",
}

MOCK_INBOX: dict[str, list[dict[str, str]]] = {
    "user@example.com": [
        {
            "id": "1",
            "from": "secretary@association.com",
            "subject": "Q4 planning meeting",
            "body": (
                "Hi, can we schedule a meeting next week? "
                "Let me know what times work for you."
            ),
        },
        {
            "id": "2",
            "from": "newsletter@shop.com",
            "subject": "Sale - 30% off",
            "body": "Don't miss our spring sale! Use code SPRING30 at checkout.",
        },
    ],
}

SENT_LOG: list[dict[str, str]] = []


def _require_auth(runtime: ToolRuntime) -> str | None:
    if not runtime.state.get("authenticated"):
        return "Error: you must log in before using this tool."
    return None


@tool
def login(
    email_address: str, password: str, runtime: ToolRuntime
) -> Command | str:
    """Authenticate the user with an email address and password.

    Returns:
        Command: State data update on successful login.
    """
    if email_address not in MOCK_USERS or MOCK_USERS[email_address] != password:
        return "Login failed: invalid email address or password."

    return Command(
        update={
            "email_address": email_address,
            "password": password,
            "authenticated": True,
            "messages": [
                ToolMessage(
                    "Successfully logged in",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def read_inbox(runtime: ToolRuntime) -> str:
    """List emails in the inbox.

    Returns:
        str: id, sender, and subject for each message in the inbox.
    """
    if error := _require_auth(runtime):
        return error

    inbox = MOCK_INBOX.get(runtime.state["email_address"], [])
    if not inbox:
        return "Inbox is empty."

    lines = "\n".join(
        f"- [{message['id']}] From: {message['from']} | Subject: {message['subject']}"
        for message in inbox
    )
    return f"Inbox:\n{lines}"


@tool
def read_email(email_id: str, runtime: ToolRuntime) -> str:
    """Read the full content of an email by its id.

    Returns:
        str: sender, subject and body of the requested message.
    """
    if error := _require_auth(runtime):
        return error

    for email in MOCK_INBOX.get(runtime.state["email_address"], []):
        if email["id"] == email_id:
            return (
                f"From: {email['from']}\n"
                f"Subject: {email['subject']}\n\n"
                f"{email['body']}"
            )

    return f"Email with id '{email_id}' was not found."


@tool
def send_email(to: str, subject: str, body: str, runtime: ToolRuntime) -> str:
    """Send an email after human approval is granted.

    Returns:
        str: confirmation message.
    """
    if error := _require_auth(runtime):
        return error

    sender = runtime.state["email_address"]
    SENT_LOG.append(
        {
            "from": sender,
            "to": to,
            "subject": subject,
            "body": body,
        }
    )
    return f"Email sent from {sender} to {to} with subject '{subject}'."
