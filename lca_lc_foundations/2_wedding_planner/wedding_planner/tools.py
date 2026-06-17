# Tools for the sub-agents:
# - Flights agent MCP: Kiwi
# - Venue agent: web search with Tavily
# - Playlist agent: DB search

import asyncio
from typing import Dict, Any

from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain_community.utilities import SQLDatabase
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from tavily import TavilyClient

import wedding_planner.agent as agents


@tool
def update_state(
    origin: str,
    destination: str,
    num_guests: str,
    genre: str,
    runtime: ToolRuntime,
) -> str:
    """Update the state when you know all of the values: origin, destination, num_guests and genre.
    This tool must be called alone, without any other tool calls. It must complete and return to make,
    the information available to other tools"""
    return Command(
        update={
            "origin": origin,
            "destination": destination,
            "num_guests": num_guests,
            "genre": genre,
            "messages": [
                ToolMessage(
                    "Successfully updated state",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


RETRYABLE_MCP_CODES = {-32603}


class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

    - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
    - Non-retryable McpError codes (e.g. -32602): return error message immediately.
    - Any other exception (fetch failed, network errors, etc.): retry then return error message.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(
                    f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                    f"(code {exc.error.code}, attempt {attempt + 1}/{self.max_retries}): {exc}"
                )
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Tool call failed (non-retryable): {exc}",
                            )
                        ],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(
                    f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                    f"(attempt {attempt + 1}/{self.max_retries}): {exc}"
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2**attempt)

        print(
            f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}"
        )
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Tool call failed after {self.max_retries} attempts: {last_error}",
                )
            ],
            isError=False,
        )


@tool
async def flight_search_mcp():
    """MCP server to search for flights to and from the destination"""
    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
            }
        },
        tool_interceptors=[RetryMCPInterceptor()],
    )

    return await client.get_tools()


@tool
async def travel_search(runtime: ToolRuntime) -> str:
    """Travel agent which searches for flights to and from the desired wedding
    location"""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    travel_agent = agents.TravelAgent()
    return await travel_agent.interact(origin, destination)


tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information using Tavily"""
    return tavily_client.search(query)


@tool
async def venue_search(runtime: ToolRuntime) -> str:
    """Venue agent chooses the best venue for the given location and number of guests (num_guests)"""
    destination = runtime.state["destination"]
    num_guests = runtime.state["num_guests"]
    venue_agent = agents.VenueAgent()
    return await venue_agent.interact(destination, num_guests)


db = SQLDatabase.from_uri(
    "sqlite:///lca_lc_foundations/2_wedding_planner/resources/Chinook.db"
)


@tool
def query_playlist_db(query: str) -> str:
    """Query the database for playlist information"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"


@tool
async def playlist_search(runtime: ToolRuntime) -> str:
    """Agent which searches for playlists suitable for a wedding and a specific
    genre"""
    genre = runtime.state["genre"]
    playlist_agent = agents.PlaylistAgent()
    return await playlist_agent.interact(genre)
