from langchain.agents import AgentState


class EmailAgentState(AgentState):
    """Credentials and auth status for the simulated email account.

    email_address: user's account address
    password: user's account password
    authenticated: is user authenticated?
    """

    email_address: str
    password: str
    authenticated: bool
