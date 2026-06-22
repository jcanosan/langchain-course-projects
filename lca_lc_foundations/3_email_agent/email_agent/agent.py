from typing import Any
from collections.abc import Callable, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Checkpointer, Command

from email_agent import tools, state


class AbstractAgent:
    """Abstract class for shared agent methods"""

    def __init__(
        self,
        system_prompt: str,
        agent_tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]]
        | None = None,
        state_schema=None,
        checkpointer: Checkpointer = None,
        middleware: Sequence[Any] | None = None,
    ):
        self._agent = create_agent(
            "ollama:gemma4:31b-cloud",
            tools=agent_tools,
            state_schema=state_schema,
            checkpointer=checkpointer,
            middleware=middleware or (),
            system_prompt=system_prompt,
        )

    @property
    def graph(self):
        return self._agent

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

    async def stream(self, input_state: dict) -> None:
        """Call the agent and stream the model tokens for one step."""
        self._interrupt = None
        async for chunk in self._agent.astream(
            input_state,
            config=self._config,
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk["type"] != "messages":
                if (
                    chunk["type"] == "updates"
                    and "__interrupt__" in chunk["data"]
                ):
                    self._interrupt = chunk["data"]["__interrupt__"][0].value
                continue

            msg_chunk, metadata = chunk["data"]
            if metadata.get("langgraph_node") == "model" and msg_chunk.content:
                print(msg_chunk.content, end="", flush=True)


class EmailAgent(AbstractAgent):
    """Email assistant that authenticates, reads inbox messages, and sends mail
    with human in the loop approval."""

    def __init__(self):
        super().__init__(
            agent_tools=[
                tools.login,
                tools.read_inbox,
                tools.read_email,
                tools.send_email,
            ],
            state_schema=state.EmailAgentState,
            checkpointer=InMemorySaver(),
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "send_email": {
                            "allowed_decisions": ["approve", "reject"],
                            "description": (
                                "Review this outgoing email before it is sent."
                            ),
                        },
                        "login": False,
                        "read_inbox": False,
                        "read_email": False,
                    },
                    description_prefix="Email action pending approval",
                )
            ],
            system_prompt=(
                """You are an email assistant for a simulated mailbox.
                Always authenticate with the login tool before reading or sending email.
                Use read_inbox to list messages and read_email to open a specific message.
                When the user asks you to reply or send mail, draft the email and call send_email.
                Never claim an email was sent unless send_email completed successfully."""
            ),
        )
        self._config = {"configurable": {"thread_id": str(uuid7())}}

    async def _prompt_for_decisions(
        self, action_requests: list[dict]
    ) -> list[dict]:
        decisions = []
        for action in action_requests:
            print("\n--- Approval required to send email ---")
            print(f"To: {action['args'].get('to')}")
            print(f"Subject: {action['args'].get('subject')}")
            print(f"Body:\n{action['args'].get('body')}")
            answer = input("Send this email? [y/n]: ").strip().lower()
            if answer in {"y", "yes"}:
                decisions.append({"type": "approve"})
            else:
                decisions.append(
                    {
                        "type": "reject",
                        "message": (
                            "User rejected sending this email. "
                            "Do not retry unless they explicitly ask again."
                        ),
                    }
                )
        return decisions

    async def interact(self, message: str):
        """Send a message to the agent so its streamed response is printed.
        Interrupts and prompts the user for a decision if paused."""
        print(f"\nUser: {message}")
        print("Assistant: ", end="", flush=True)
        await self.stream({"messages": [HumanMessage(content=message)]})
        while self._interrupt:
            decisions = await self._prompt_for_decisions(
                self._interrupt["action_requests"]
            )
            await self.stream(Command(resume={"decisions": decisions}))
        print("\n\n=== === ===")
