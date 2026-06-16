# Personal chef assistant agent

from pathlib import Path

from dotenv import load_dotenv
from langchain.messages import HumanMessage

from personal_chef.chef_assistant_agent import (
    ChefAssistantAgentTextBased,
    ChefAssistantAgentImageBased,
)

load_dotenv()

chef_assistant_agent = ChefAssistantAgentTextBased()
chef_assistant_agent.send_message(
    HumanMessage(content="I have eggs, potatoes and an onion. I'm poor.")
)
chef_assistant_agent.send_message(
    HumanMessage(content="How should I cut the potatoes for the potato omelet?")
)

print("\n\n=============\n\n")

chef_assistant_agent_images = ChefAssistantAgentImageBased()


def get_image_path():
    base_path = Path(__file__).parent
    return base_path / "assets" / "pantry.jpg"


chef_assistant_agent_images.send_image(image_path=get_image_path())
chef_assistant_agent_images.send_message(
    HumanMessage(content="I actually also have eggs. Could I make a frittata?")
)
