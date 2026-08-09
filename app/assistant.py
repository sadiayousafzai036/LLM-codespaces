"""Wiring: database -> knowledge base -> retriever -> assistant.

Also a command-line entry point, which is the quickest way to check that the
whole chain works after ingestion:

    uv run python -m app.assistant "why is my sourdough so dense?"
"""

import sys

from app import db, settings
from app.llm import LLMClient
from app.rag import CookingAssistant
from app.retrieval import RetrieverFactory


def load_knowledge_base() -> list[dict]:
    documents = db.load_documents()
    if not documents:
        raise RuntimeError(
            "the documents table is empty - run `make ingest` first"
        )
    return documents


def build_assistant(strategy: str | None = None, prompt_variant: str | None = None,
                    model: str | None = None, documents: list[dict] | None = None,
                    show_progress: bool = True) -> CookingAssistant:
    documents = documents if documents is not None else load_knowledge_base()
    llm = LLMClient()
    factory = RetrieverFactory(documents, llm=llm, show_progress=show_progress)
    retriever = factory.build(strategy or settings.RETRIEVAL_STRATEGY)
    return CookingAssistant(
        retriever=retriever, llm=llm, prompt_variant=prompt_variant, model=model
    )


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('usage: python -m app.assistant "your cooking question"')
        raise SystemExit(1)

    assistant = build_assistant()
    result = assistant.answer(question)

    print()
    print(result.answer)
    print()
    print(f"strategy: {result.strategy} | prompt: {result.prompt_variant} "
          f"| model: {result.model}")
    print(f"retrieval {result.retrieval_time:.2f}s | total {result.response_time:.2f}s "
          f"| {result.usage.total_tokens} tokens")
    if result.rewritten_query:
        print(f"rewritten query: {result.rewritten_query}")
    print("\nsources:")
    for number, document in enumerate(result.documents, start=1):
        print(f"  [{number}] {document['title']} - {document['link']}")

    conversation_id = db.save_conversation(result.as_conversation_row())
    print(f"\nlogged as conversation {conversation_id}")


if __name__ == "__main__":
    main()
