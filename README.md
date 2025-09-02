## RAG System Methodology: Key Points

To create a highly accurate information retrieval system for dense financial PDFs (like SEC filings)

### System Architecture

Phase 1: Offline Indexing:

1. Load PDFs and extract text.

2. Extract SEC Item numbers (e.g., "Item 1A. Risk Factors") as metadata for context.

3. Chunk text semantically using RecursiveCharacterTextSplitter.

4. Embed chunks with Google's Gemini model.

5. Store in a ChromaDB vector database.

Phase 2: Online Querying:

1. Fetch a large set of candidate documents via initial vector search.

2. Apply an Advanced 3-Stage Reranking process to refine results.

3. Pass the top-ranked chunks as context to a Gemini LLM.

4. Generate a final answer with verifiable source citations.

### 3-Stage Hybrid Reranking
This multi-step process ensures both semantic relevance and keyword precision.

Stage 1: Keyword Filtering & Scoring

Why: To catch documents with exact term matches (like "$1.5 million" or "liabilities") that pure semantic search might miss.

How: Scores documents using a custom financial vocabulary, protected keywords, and pattern matching (e.g., $, %, years).

Stage 2: Cross-Encoder Reranking

Why: To get a much more accurate semantic relevance score than the initial vector search.

How: Uses FlashRank, a cross-encoder that evaluates the query and document together, providing superior contextual understanding.

Stage 3: Hybrid Scoring

Why: To get the best of both worlds.

How: The final rank is a weighted combination of the keyword score (for precision) and the cross-encoder score (for semantic nuance).

### Generation & Prompt Engineering
Preventing Hallucinations: A strong system prompt strictly commands the LLM to answer only from the provided document chunks.

Ensuring Verifiability: The prompt enforces a structured JSON output ({"answer": "...", "ref_ids": [...]}), which guarantees that every answer is tied directly back to a specific source chunk (CHUNK_ID).

### Future Improvements
1. Add query transformation to handle more complex, multi-part questions.
2. use painted html to pass images to catch the highlighted text or texts with larger fonts showcasing important information.


Note: Idea for different techniques has been taken from different sources and works done on SEC filings dataset and problems