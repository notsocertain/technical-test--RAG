import os
import sys

import json

# Fix system path by appending the parent directory of tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from pipeline import Generationclass


def run_tests():
    """Runs a series of tests on the generationclass."""

    # 1. Instantiate the class
    print("Instantiating generationclass...")
    try:
        gen_client = Generationclass()
        print("Success: generationclass instantiated.")
    except ValueError as e:
        print(f"Error: {e}. Please set your GEMINI_API_KEY environment variable.")
        return

    # 2. Test with relevant context
    print("\n--- Test Case 1: Relevant Context Provided ---")
    relevant_query = "What is the capital of France?"
    mock_results_1 = [
        {
            "chunk_id": "doc_abc.pdf|p1|c0",
            "content": "Paris is the capital and most populous city of France.",
            "document_name": "france_info.pdf",
            "page_number": 1,
            "item_number": "3",
            "item_title": "Geography",
            "rerank_score": 0.95,
        },
        {
            "chunk_id": "doc_abc.pdf|p5|c2",
            "content": "The Eiffel Tower is a landmark in Paris, France.",
            "document_name": "france_info.pdf",
            "page_number": 5,
            "item_number": "5",
            "item_title": "Landmarks",
            "rerank_score": 0.88,
        },
    ]

    response_1 = gen_client.generate(relevant_query, mock_results_1)
    print("Query:", relevant_query)
    print("Response:", json.dumps(response_1, indent=2))
    print("--- Expected Output ---")
    print("  - An answer derived from the first chunk.")
    print("  - A 'sources' list with 'France Info' and 'p. 2'.")
    print("-" * 50)

    # 3. Test with irrelevant context
    print("\n--- Test Case 2: Irrelevant Context Provided ---")
    irrelevant_query = "What is the average rainfall in the Amazon?"
    mock_results_2 = [
        {
            "chunk_id": "doc_def.pdf|p10|c0",
            "content": "The company's revenue for Q3 2024 was $50 million.",
            "document_name": "company_report.pdf",
            "page_number": 10,
            "item_number": "7",
            "item_title": "Financials",
            "rerank_score": 0.70,
        },
    ]

    response_2 = gen_client.generate(irrelevant_query, mock_results_2)
    print("Query:", irrelevant_query)
    print("Response:", json.dumps(response_2, indent=2))
    print("--- Expected Output ---")
    print(f"  - The answer should be 'not found message'.")
    print("  - The 'sources' list should be empty.")
    print("-" * 50)


# Run the tests
if __name__ == "__main__":
    run_tests()
