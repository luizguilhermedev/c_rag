from langgraph.prebuilt import create_react_agent
from app.application.tools.retrieve_tool import query_rewrite_tool, retriever_tool, grounded_or_not_tool
from langgraph.checkpoint.memory import MemorySaver

from app.domain.prompts.react_agent_service_prompt import react_agent_prompt
from app.infrastructure.initialize_llm import initialize_llm

tools = [retriever_tool, query_rewrite_tool, grounded_or_not_tool]

checkpointer = MemorySaver()

class ReactAgent:
    def __init__(self):
        self.agent = create_react_agent(
            model=initialize_llm(),
            tools=tools,
            checkpointer=checkpointer,
            prompt=react_agent_prompt,
        )

    def stream(self, input, config):
        for s in self.agent.stream(input, config, stream_mode="values"):
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

if __name__ == "__main__":
    inputs = {"messages": [("user", "What make Darwin to write th Origin of the species?")]}
    config = {"configurable": {"thread_id": "thread-10"}}
    agent = ReactAgent()
    agent.stream(inputs, config)