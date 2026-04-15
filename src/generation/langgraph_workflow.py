"""LangChain + LangGraph orchestration for Stage 6 generation."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.generation.prompt_templates import SYSTEM_PROMPT, build_user_prompt


class GenerationState(TypedDict):
    query: str
    clauses: list[dict[str, Any]]
    tool_results: dict[str, Any]
    chat_history: list[dict[str, str]]
    prompt: str
    raw_output: str
    parsed_output: dict[str, Any]


class GenerationWorkflow:
    """State-graph workflow that builds prompt, generates output, and validates JSON."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 320,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        text_gen = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
        )

        self.llm = HuggingFacePipeline(pipeline=text_gen)
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("user", "{user_prompt}"),
            ])
            | self.llm
            | StrOutputParser()
        )
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GenerationState)
        graph.add_node("prepare_prompt", self._prepare_prompt)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate)

        graph.set_entry_point("prepare_prompt")
        graph.add_edge("prepare_prompt", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", END)
        return graph

    @staticmethod
    def _prepare_prompt(state: GenerationState) -> dict[str, Any]:
        prompt = build_user_prompt(
            query=state["query"],
            clauses=state.get("clauses", []),
            tool_results=state.get("tool_results", {}),
            chat_history=state.get("chat_history", []),
        )
        return {"prompt": prompt}

    def _generate(self, state: GenerationState) -> dict[str, Any]:
        raw = self.chain.invoke({"user_prompt": state["prompt"]})
        return {"raw_output": raw}

    @staticmethod
    def _validate(state: GenerationState) -> dict[str, Any]:
        from src.generation.generator import ContractGenerator

        parsed = ContractGenerator._safe_parse_json(state.get("raw_output", ""))
        return {"parsed_output": parsed}

    def run(
        self,
        query: str,
        clauses: list[dict[str, Any]],
        tool_results: dict[str, Any] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        state: GenerationState = {
            "query": query,
            "clauses": clauses,
            "tool_results": tool_results or {},
            "chat_history": chat_history or [],
            "prompt": "",
            "raw_output": "",
            "parsed_output": {},
        }
        output = self.graph.invoke(state)
        return output["parsed_output"]
