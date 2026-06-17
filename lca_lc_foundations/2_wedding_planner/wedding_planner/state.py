from langchain.agents import AgentState


class WeddingState(AgentState):
    """Necessary data to search in all agents
    origin: city of origin
    destination: city where the wedding will happen
    num_guests: number of guests
    genre: playlist genre
    """

    origin: str
    destination: str
    num_guests: str
    genre: str
