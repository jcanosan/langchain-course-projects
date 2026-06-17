# Wedding planner agent
# Multi-agent system with a main coordinator and sub-agents (tools):
# - Flights agent: to and from the destination
# - Venue agent: searches the web for a wedding venue
# - Playlist agent: DJ agent that goes through a music DB for a playlist
# matching a specific genre

import asyncio
from dotenv import load_dotenv

from wedding_planner.agent import WeddingCoordinator


async def main():
    load_dotenv()
    wedding_coordinator_agent = WeddingCoordinator()
    await wedding_coordinator_agent.interact(
        "I'm from Amsterdam and I'd like a wedding in Helsinki for 100 guests. "
        "For the playlist, my wedding is going to have a goth/black metal theme"
    )


if __name__ == "__main__":
    asyncio.run(main())
