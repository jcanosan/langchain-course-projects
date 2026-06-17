# Multi-agent system
# - Main coordinator agent
# - Sub-agents:
#   - Flights agent: to and from the destination
#   - Venue agent: searches the web for a wedding venue (number of guests!)
#   - Playlist agent: DJ agent that goes through a music DB for a playlist

from typing import Any
from uuid import uuid4
from collections.abc import Callable, Sequence

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Checkpointer

from wedding_planner import tools, state


class AbstractAgent:
    """Abstract class for shared agent methods"""

    def __init__(
        self,
        system_prompt: str,
        agent_tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]]
        | None = None,
        state_schema=None,
        checkpointer: Checkpointer = None,
    ):
        self._agent = create_agent(
            "ollama:gemma4:31b-cloud",
            tools=agent_tools,
            state_schema=state_schema,
            checkpointer=checkpointer,
            system_prompt=system_prompt,
        )

    async def interact(
        self,
        message: HumanMessage,
    ):
        response = await self._agent.ainvoke(
            {"messages": message},
            config=self._config,
            allowed_objects="messages",
        )
        return response["messages"][-1].content

    async def stream(
        self,
        message: HumanMessage,
    ):
        async for msg, metadata in self._agent.astream(
            {"messages": message},
            config=self._config,
            stream_mode="messages",
        ):
            yield msg, metadata


class WeddingCoordinator(AbstractAgent):
    """Wedding agent which coordinates the sub-agents"""

    def __init__(self):
        super().__init__(
            agent_tools=[
                tools.update_state,
                tools.travel_search,
                tools.venue_search,
                tools.playlist_search,
            ],
            state_schema=state.WeddingState,
            checkpointer=InMemorySaver(),
            system_prompt=(
                """You are a wedding coordinator.
                First find all the information you need to update the state. When you have the information, update the state.
                Once that has completed and returned, you can delegate the tasks to your specialists for flights, venues and a musical playlist.
                Once you have received their answers, coordinate the perfect wedding for me"""
            ),
        )
        self._config = {"configurable": {"thread_id": str(uuid4)}}

    async def interact(self, message: str):
        """Send human message to the agent and only print its streamed response.
        Tool messages are ignoredS

        Args:
            message (str): human utterance for the agent
        """

        print(f"User: {message}")
        print("System: ", end="")
        async for sys_chunk, metadata in self.stream(
            message=HumanMessage(content=message)
        ):
            if metadata.get("langgraph_node") == "model":
                print(sys_chunk.content, end="", flush=True)


class TravelAgent(AbstractAgent):
    """Travel search sub-agent"""

    def __init__(self):
        super().__init__(
            agent_tools=[tools.flight_search_mcp],
            system_prompt=(
                """You are a travel agent. Search for flights to the desired destination wedding location.
                You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
                - Price (lowest, economy class)
                - Duration (shortest)
                - Date (time of year which you believe is best for a wedding at this location)
                To make things easy, only look for one ticket, one way.
                You may need to make multiple searches to iteratively find the best options.
                You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
                If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
                Once you have found the best options, let the user know your shortlist of options"""
            ),
        )
        self._config = None

    async def interact(self, origin: str, destination: str):
        response = await super().interact(
            message=HumanMessage(
                content=f"Find flights from {origin} to {destination}"
            )
        )
        return response


class VenueAgent(AbstractAgent):
    """Sub-agent which searches the web for a wedding venue"""

    def __init__(self):
        super().__init__(
            agent_tools=[tools.web_search],
            system_prompt=(
                """You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
                You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
                - Price (lowest)
                - Capacity (exact match)
                - Reviews (highest)
                You may need to make multiple searches to iteratively find the best options. 
                You have a suggested limit of 12 web searches. Count every web_search call you make.
                After 12 searches, you should stop searching and summarize the best options you have
                found so far."""
            ),
        )
        self._config = None

    async def interact(self, destination: str, capacity: str):
        response = await super().interact(
            message=HumanMessage(
                content=f"Find wedding venues in {destination} for {capacity} guests"
            )
        )
        return response


class PlaylistAgent(AbstractAgent):
    """Sub-agent which searches the web for a wedding venue"""

    def __init__(self):
        super().__init__(
            agent_tools=[tools.query_playlist_db],
            system_prompt=(
                """You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
                Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
                If you run into errors when querying the database, try to fix them by making changes to the query.
                Do not come back empty handed, keep trying to query the db until you find a list of songs.
                This is a SQLite database. Before writing any data queries, first discover the schema."""
            ),
        )
        self._config = None

    async def interact(self, genre: str):
        response = await super().interact(
            message=HumanMessage(
                content=f"Find a wedding playlist for this genre: {genre}"
            )
        )
        return response
