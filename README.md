# Description

Project scripts for the LangChain courses

## Projects summary

### Personal chef assistant agent

Searches the web using [Tavily](https://www.tavily.com/) for recipes that can be
made with the ingredients which may be provided by the user via:

- Text: a list of ingredients.
- Image: an image containing ingredients. E.g. a photo of a pantry.

Streams recipe suggestions back to the user. Memory is integrated so the system
can answer follow-up questions.

### Wedding planner agent

Multi-agent system with a main coordinator and sub-agents (tools):

- Flights agent: uses the [Kiwi Travel MCP server](https://mcp.so/server/kiwi-travel-mcp/Vytautas%20Dargis)
  to search for flights and from the destination.
- Venue agent: searches the web for a wedding venue using [Tavily](https://www.tavily.com/).
- Playlist agent: DJ agent that goes through the [Chinook database](https://github.com/lerocha/chinook-database)
  for a playlist matching a specific genre.

### Email agent

A dummy email agent which simulates interacting with an email account.

- Authenticates a user.
- Reads the user's inbox content.
- Reads an email from the inbox.
- Sends emails after human approval is provided.
