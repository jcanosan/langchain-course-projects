# Email agent
# Simulated mailbox with login, inbox reading, and human-approved sending

import asyncio

from dotenv import load_dotenv

from email_agent.agent import EmailAgent

email_agent = EmailAgent().graph


async def main():
    load_dotenv()
    agent = EmailAgent()
    await agent.interact(
        "Log me in with user@example.com and password password123, "
        "then show my inbox."
    )
    await agent.interact(
        "Read the first email"
    )
    await agent.interact(
        "Tuesday at 2pm works for me."
    )


if __name__ == "__main__":
    asyncio.run(main())
