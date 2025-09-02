from utils.questions import Questions
from pipeline import answer_question
from pipeline import Chromaclass, Generationclass
from params import TOP_K, NOT_FOUND_MSG


def answer_question(query: str) -> dict:
    """
    Answers a question using the complete RAG pipeline.
    """
    vecstore = Chromaclass()
    results = vecstore.retrieve_enhanced(query, k=TOP_K, prefetch=30 * 2)
    generator = Generationclass()
    output = generator.generate(query, results)
    return {
        "answer": output.get("answer", NOT_FOUND_MSG),
        "sources": output.get("sources", []),
    }


for q in Questions:
    result = answer_question(q["question"])
    print({"question_id": q["question_id"], "question": q["question"], **result})
