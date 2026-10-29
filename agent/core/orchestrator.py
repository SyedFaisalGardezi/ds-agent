from __future__ import annotations

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from agent.core.llm_router import router
from agent.core.memory import AgentMemory
from agent.core.tool_registry import ToolRegistry

SYSTEM_PROMPT = """You are a senior data scientist agent. You have access to the following tools:

{tools}

Use this format:
Thought: reason about the task
Action: tool name
Action Input: input to the tool
Observation: tool result
... (repeat as needed)
Thought: I now know the final answer
Final Answer: the answer

Question: {input}
{agent_scratchpad}"""


class DSOrchestrator:
    def __init__(self) -> None:
        self.memory = AgentMemory()
        self._executor: AgentExecutor | None = None

    def _build(self) -> AgentExecutor:
        llm = router.get("plan")
        tools = ToolRegistry.get().all()
        prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory.short_term,
            verbose=True,
            handle_parsing_errors=True,
        )

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = self._build()
        return self._executor

    def run(self, task: str) -> str:
        return self.executor.invoke({"input": task})["output"]
