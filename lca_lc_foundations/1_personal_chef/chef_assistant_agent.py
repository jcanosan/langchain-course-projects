# Agent objects

import base64

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agent_tools as tools


class ChefAssistantAgent:
    """Personal chef assistant agent that:
    1. Takes a list (or image) of ingredients from the user
    2. Searches the web for recipes that can be made with those ingredients
    3. Returns recipe suggestions to the user
    4. Can answer follow-up questions (has memory!)
    """

    def __init__(self, system_prompt: str):
        self._agent = create_agent(
            "ollama:gemma4:31b-cloud",
            tools=[tools.web_search],
            checkpointer=InMemorySaver(),
            system_prompt=system_prompt,
        )

    def send_message(
        self,
        message: HumanMessage,
    ):
        response = self._agent.invoke(
            {"messages": message},
            config=self._config,
            allowed_objects="messages",
        )
        print(response["messages"][-1].content)


class ChefAssistantAgentTextBased(ChefAssistantAgent):
    def __init__(self):
        super().__init__(
            system_prompt="You are a personal chef assistant. "
            "A user will give you a list of ingredients."
            "Based on those, you search the web for recipes that can be done."
            "The user may ask you follow-up questions."
        )
        self._config = {"configurable": {"thread_id": "1"}}


class ChefAssistantAgentImageBased(ChefAssistantAgent):
    def __init__(self):
        super().__init__(
            system_prompt="You are a personal chef assistant. "
            "A user will give you an image containing some ingredients."
            "Based on those, you search the web for recipes that can be done."
            "The user may ask you follow-up questions."
        )
        self._config = {"configurable": {"thread_id": "2"}}

    def open_image(self, image_path):
        with open(image_path, "rb") as image_file:
            self._image = base64.b64encode(image_file.read()).decode("utf-8")

    def send_image(self, image_path):
        self.open_image(image_path)

        multimodal_message = HumanMessage(
            content=[
                {
                    "type": "image",
                    "base64": self._image,
                    "mime_type": "image/jpg",
                },
            ]
        )
        response = self._agent.invoke(
            {"messages": [multimodal_message]},
            config=self._config,
            allowed_objects="messages",
        )
        print(response["messages"][-1].content)
