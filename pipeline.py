import warnings

warnings.filterwarnings("ignore")


import os, os.path as op, glob
import re
import json
from typing import List, Tuple, Dict, Any


from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

from flashrank import Ranker, RerankRequest
from groq import Groq

import re
import nltk
from typing import List, Dict, Set, Tuple

from params import (
    EMBED_MODEL,
    FLASHRANK_MODEL,
    COLLECTION_NAME,
    NOT_FOUND_MSG,
    GENERATIVE_MODEL,
    TOP_K,
    PREFETCH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_DIR,
    CHROMA_DB_PATH,
    GEMINI_API_KEY,
    FINANCE_STOPWORD,
)

from utils.prompt import USER_PROMPT_TEMPLATE, SYSTEM_PROMPT
from utils.questions import questions


try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
print("NLTK data downloaded")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


import google.generativeai as genai

# Set your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)


class Reranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.ranker = Ranker(model_name=model_name)

        # Extended stopwords for financial documents
        self.english_stopwords = set(stopwords.words("english"))

        # Important financial/SEC keywords
        self.protected_keywords = FINANCE_STOPWORD

    def _clean_and_tokenize(self, text: str) -> List[str]:
        """Clean text and tokenize while preserving important terms"""

        text = text.lower()

        text = re.sub(r"[^\w\s\$%]", " ", text)

        tokens = word_tokenize(text)

        cleaned_tokens = []
        for token in tokens:
            if token in self.protected_keywords:
                cleaned_tokens.append(token)
            elif re.match(r"^\d+(?:\.\d+)?$", token):
                cleaned_tokens.append(token)
            elif "$" in token or "%" in token:
                cleaned_tokens.append(token)
            elif token not in self.english_stopwords and len(token) > 2:
                cleaned_tokens.append(token)

        return cleaned_tokens

    def _extract_financial_patterns(self, text: str) -> List[str]:
        """Extract specific financial patterns"""
        patterns = []

        # Dollar amounts
        dollar_patterns = re.findall(
            r"\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?", text.lower()
        )
        patterns.extend(dollar_patterns)

        # Percentages
        percent_patterns = re.findall(r"\d+(?:\.\d+)?%", text)
        patterns.extend(percent_patterns)

        # Years
        year_patterns = re.findall(r"\b(?:19|20)\d{2}\b", text)
        patterns.extend(year_patterns)

        # SEC items
        item_patterns = re.findall(r"\bitem\s+\d+[a-z]?\b", text.lower())
        patterns.extend(item_patterns)

        return patterns

    def _calculate_keyword_similarity(
        self,
        query_tokens: List[str],
        doc_tokens: List[str],
        query_patterns: List[str],
        doc_patterns: List[str],
    ) -> Dict[str, float]:
        """Calculate various keyword similarity metrics"""

        # 1. Token overlap score
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)

        if not query_set:
            return {
                "token_overlap": 0.0,
                "pattern_match": 0.0,
                "jaccard": 0.0,
                "weighted_score": 0.0,
            }

        intersection = query_set.intersection(doc_set)
        token_overlap = len(intersection) / len(query_set)

        # 2. Financial pattern matching
        pattern_matches = 0
        for pattern in query_patterns:
            if pattern in doc_patterns:
                pattern_matches += 1

        pattern_match = pattern_matches / max(1, len(query_patterns))

        # 3. Jaccard similarity
        union = query_set.union(doc_set)
        jaccard = len(intersection) / len(union) if union else 0.0

        # 4. Weighted score
        weighted_intersection = 0
        for token in intersection:
            if token in self.protected_keywords:
                weighted_intersection += 2.0
            else:
                weighted_intersection += 1.0

        weighted_score = weighted_intersection / len(query_set)

        return {
            "token_overlap": token_overlap,
            "pattern_match": pattern_match,
            "jaccard": jaccard,
            "weighted_score": weighted_score,
        }

    def _filter_by_keywords(
        self, query: str, docs: List[Document], min_similarity: float = 0.1
    ) -> List[Tuple[Document, Dict[str, float]]]:
        """Filter and score documents based on keyword similarity"""

        query_tokens = self._clean_and_tokenize(query)
        query_patterns = self._extract_financial_patterns(query)

        scored_docs = []

        for doc in docs:
            doc_tokens = self._clean_and_tokenize(doc.page_content)
            doc_patterns = self._extract_financial_patterns(doc.page_content)

            similarity_metrics = self._calculate_keyword_similarity(
                query_tokens, doc_tokens, query_patterns, doc_patterns
            )

            combined_score = (
                0.4 * similarity_metrics["weighted_score"]
                + 0.3 * similarity_metrics["token_overlap"]
                + 0.2 * similarity_metrics["pattern_match"]
                + 0.1 * similarity_metrics["jaccard"]
            )

            if combined_score >= min_similarity:
                scored_docs.append(
                    (
                        doc,
                        {
                            **similarity_metrics,
                            "combined_keyword_score": combined_score,
                        },
                    )
                )

        scored_docs.sort(key=lambda x: x[1]["combined_keyword_score"], reverse=True)

        return scored_docs

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 5,
        keyword_filter_threshold: float = 0.1,
        hybrid_weight: float = 0.3,
    ) -> List[Document]:
        """
        Three-stage reranking:
        1. Remove stopwords and filter by keyword similarity
        2. Apply FlashRank to filtered documents
        3. Combine keyword + FlashRank scores
        """
        if not docs:
            return []

        # print(f"[pipeline] Starting 3-stage reranking with {len(docs)} documents")

        keyword_scored_docs = self._filter_by_keywords(
            query, docs, keyword_filter_threshold
        )

        if not keyword_scored_docs:
            # print("[pipeline] No documents passed keyword filter, using all documents")
            keyword_scored_docs = [
                (doc, {"combined_keyword_score": 0.0}) for doc in docs
            ]

        candidates = keyword_scored_docs
        candidate_docs = [doc for doc, _ in candidates]

        # print(f"[pipeline] Stage 1 complete: {len(candidates)} candidates for FlashRank")
        try:
            passages = [{"text": d.page_content} for d in candidate_docs]
            flashrank_results = self.ranker.rerank(
                RerankRequest(query=query, passages=passages)
            )

            final_results = []

            for i, flashrank_item in enumerate(flashrank_results):
                idx = i

                if idx >= len(candidates):
                    continue

                doc, keyword_metrics = candidates[idx]
                flashrank_score = flashrank_item.get("score", 0.0)
                normalized_flashrank = (flashrank_score + 1) / 2
                keyword_score = keyword_metrics["combined_keyword_score"]
                final_score = (
                    1 - hybrid_weight
                ) * normalized_flashrank + hybrid_weight * keyword_score

                final_results.append(
                    {
                        "doc": doc,
                        "final_score": final_score,
                        "flashrank_score": flashrank_score,
                        "keyword_score": keyword_score,
                        "keyword_metrics": keyword_metrics,
                    }
                )
            final_results.sort(key=lambda x: x["final_score"], reverse=True)

            reranked_docs = []
            for i, result in enumerate(final_results[:top_k]):
                doc = result["doc"]
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata["rerank_score"] = result["final_score"]
                enhanced_metadata["flashrank_score"] = result["flashrank_score"]
                enhanced_metadata["keyword_score"] = result["keyword_score"]
                enhanced_metadata.update(result["keyword_metrics"])

                reranked_docs.append(
                    Document(page_content=doc.page_content, metadata=enhanced_metadata)
                )

            return reranked_docs

        except Exception as e:
            print(f"Exception Occured while reranking: {e}")
            # Fallback: Use only keyword scores
            fallback_docs = []
            for i, (doc, metrics) in enumerate(candidates[:top_k]):
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata["rerank_score"] = metrics["combined_keyword_score"]
                enhanced_metadata["keyword_score"] = metrics["combined_keyword_score"]
                enhanced_metadata.update(metrics)

                fallback_docs.append(
                    Document(page_content=doc.page_content, metadata=enhanced_metadata)
                )

            return fallback_docs


def retrieve_sec_info_enhanced(
    vectorstore: Chroma, query: str, k: int = 5, prefetch: int = PREFETCH
) -> List[Dict]:
    """Enhanced retrieval with stopword removal and keyword matching"""
    print(f"[retrieve] Enhanced query: {query}")

    candidates = vectorstore.similarity_search(query, k=prefetch)  # Get more candidates
    print(f"[retrieve] Retrieved {len(candidates)} initial candidates")

    enhanced_reranker = Reranker("ms-marco-MiniLM-L-12-v2")

    top_docs = enhanced_reranker.rerank(
        query, candidates, top_k=k, keyword_filter_threshold=0.05, hybrid_weight=0.4
    )

    print(f"[retrieve] Final reranked results: {len(top_docs)}")

    results = []
    for doc in top_docs:
        metadata = doc.metadata or {}
        results.append(
            {
                "chunk_id": metadata.get("chunk_id", ""),
                "content": doc.page_content,
                "document_name": metadata.get("document_name", ""),
                "page_number": metadata.get("page", 0),
                "item_number": metadata.get("item_number", ""),
                "item_title": metadata.get("item_title", ""),
                "rerank_score": metadata.get("rerank_score", 0.0),
                # Additional keyword metrics
                "keyword_score": metadata.get("keyword_score", 0.0),
                "flashrank_score": metadata.get("flashrank_score", 0.0),
                "token_overlap": metadata.get("token_overlap", 0.0),
                "pattern_match": metadata.get("pattern_match", 0.0),
            }
        )

    return results


class Embeddingclass(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL, load_model: bool = True):
        self.model_name = model_name
        self.load_model = load_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents using the Google Gemini embedding model.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of lists of floats, where each inner list is the embedding for a document.
        """

        # Google's `embed_content` can handle a list of strings directly,
        # so no need for manual batching unless you hit API limits.
        try:
            result = genai.embed_content(model=self.model_name, content=texts)
            return result["embedding"]
        except Exception as e:
            print(f"An error occurred while embedding documents: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string using the Google Gemini embedding model.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding of the query.
        """
        try:
            result = genai.embed_content(model=self.model_name, content=[text])
            # The result is a list containing a single embedding, so we return the first element.
            return result["embedding"][0]
        except Exception as e:
            print(f"An error occurred while embedding the query: {e}")
            return []


class rerankerclass:
    def __init__(self, model_name: str = FLASHRANK_MODEL):
        self.ranker = Ranker(model_name=model_name)

    def rerank(
        self, query: str, docs: List[Document], top_k: int = 5
    ) -> List[Document]:
        if not docs:
            return []
        try:
            passages = [{"text": d.page_content} for d in docs]
            results = self.ranker.rerank(RerankRequest(query=query, passages=passages))
            reranked = []
            for item in results[:top_k]:
                idx = getattr(item, "index", getattr(item, "corpus_id", None))
                if idx is None and isinstance(item, dict):
                    idx = item.get("index", item.get("corpus_id", 0))
                if idx is not None and idx < len(docs):
                    doc = docs[idx]
                    md = dict(doc.metadata or {})
                    score = (
                        float(getattr(item, "score", item.get("score", 0.0)))
                        if isinstance(item, dict)
                        else float(getattr(item, "score", 0.0))
                    )
                    md["rerank_score"] = score
                    reranked.append(
                        Document(page_content=doc.page_content, metadata=md)
                    )
            return reranked
        except Exception:
            out = []
            for doc in docs[:top_k]:
                md = dict(doc.metadata or {})
                md["rerank_score"] = 0.0
                out.append(Document(page_content=doc.page_content, metadata=md))
            return out


class Generationclass:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")

        # Using LangChain's wrapper for the Gemini API
        self.client = ChatGoogleGenerativeAI(
            model=GENERATIVE_MODEL,
            google_api_key=self.api_key,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    @staticmethod
    def _coerce_and_parse_json(raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        s = raw.strip()
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.replace(""","\"").replace(""", '"').replace("'", "'").replace("'", "'")
        b0, b1 = s.find("{"), s.rfind("}")
        if b0 != -1 and b1 != -1 and b1 > b0:
            s = s[b0 : b1 + 1]
        s = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        m_ans = re.search(r'"answer"\s*:\s*"([^"]*)"', s)
        m_ids = re.search(r'"ref_ids"\s*:\s*\[(.*?)\]', s, flags=re.DOTALL)
        ans = m_ans.group(1).strip() if m_ans else ""
        ref_ids = re.findall(r'"([^"]+)"', m_ids.group(1)) if m_ids else []
        return {"answer": ans, "ref_ids": ref_ids} if (ans or ref_ids) else {}

    def _format_context(self, results: List[Dict]) -> str:
        blocks = []
        for r in results:
            cid = r.get("chunk_id", "")
            doc = r.get("document_name", "")
            page = r.get("page_number", "")
            item = r.get("item_number", "")
            header = f"[CHUNK_ID:{cid}] [DOC:{doc}] [PAGE:{page}] [ITEM:{item}]"
            txt = r["content"][:CHUNK_SIZE] + (
                "..." if len(r["content"]) > CHUNK_SIZE else ""
            )
            blocks.append(f"{header}\n{txt}")
        return "\n\n".join(blocks)

    def _chunk_ids_to_sources_flat(
        self, ref_ids: List[str], results: List[Dict]
    ) -> List[str]:
        index = {r["chunk_id"]: r for r in results if r.get("chunk_id")}
        out: List[str] = []
        for rid in ref_ids:
            r = index.get(rid)
            if not r:
                continue
            label = r.get("document_name", "")
            page = int(r.get("page_number", 0)) + 1
            item = r.get("item_number") or ""
            out.extend([label, f"Item {item}" if item else "Item", f"p. {page}"])
            break
        return out

    def generate(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {"answer": NOT_FOUND_MSG, "sources": []}

        context = self._format_context(results)
        valid_ids = [r.get("chunk_id", "") for r in results if r.get("chunk_id")]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            not_found_msg=NOT_FOUND_MSG,
            valid_ids=", ".join(valid_ids),
            query=query,
            context=context,
        )
        # print(user_prompt)

        completion = self.client.chat.completions.create(
            model=GENERATIVE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=576,
        )

        raw = (completion.choices[0].message.content or "").strip()
        obj = self._coerce_and_parse_json(raw)
        answer = (obj.get("answer") or "").strip()
        ref_ids = [
            rid
            for rid in (obj.get("ref_ids") or [])
            if rid in set([r.get("chunk_id", "") for r in results])
        ]

        if not answer or answer == NOT_FOUND_MSG:
            return {"answer": NOT_FOUND_MSG, "sources": []}

        return {
            "answer": answer,
            "sources": self._chunk_ids_to_sources_flat(ref_ids, results),
        }

    def generate(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {"answer": NOT_FOUND_MSG, "sources": []}

        context = self._format_context(results)
        valid_ids = [r.get("chunk_id", "") for r in results if r.get("chunk_id")]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            not_found_msg=NOT_FOUND_MSG,
            valid_ids=", ".join(valid_ids),
            query=query,
            context=context,
        )

        # Build messages list in a format compatible with both Groq and LangChain's LLM
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(not_found_msg=NOT_FOUND_MSG),
            },
            {"role": "user", "content": user_prompt},
        ]

        # Use the `invoke` method of the ChatGoogleGenerativeAI client
        try:
            completion = self.client.invoke(messages)
            raw = (completion.content or "").strip()
        except Exception as e:
            print(f"An error occurred during LLM generation: {e}")
            return {"answer": "An error occurred during generation.", "sources": []}

        obj = self._coerce_and_parse_json(raw)
        answer = (obj.get("answer") or "").strip()
        ref_ids = [
            rid
            for rid in (obj.get("ref_ids") or [])
            if rid in set([r.get("chunk_id", "") for r in results])
        ]

        if not answer or answer == NOT_FOUND_MSG:
            return {"answer": NOT_FOUND_MSG, "sources": []}

        return {
            "answer": answer,
            "sources": self._chunk_ids_to_sources_flat(ref_ids, results),
        }


class Chromaclass:
    _vectorstore = None

    def __init__(
        self,
        pdf_dir: str = PDF_DIR,
        collection_name: str = COLLECTION_NAME,
        persist_dir: str = CHROMA_DB_PATH,
    ):
        self.pdf_dir = pdf_dir or os.environ.get("PDF_DIR", "/content")
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        if Chromaclass._vectorstore is None:
            self._ensure_vectorstore()

    @staticmethod
    def _sec_item_patterns() -> List[str]:
        """
        Generic SEC filing pattern regex
        """
        return [
            r"Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
            r"ITEM\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
            r"Part\s+([IV]+)\s*[,\-]\s*Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
        ]

    @classmethod
    def _extract_sec_items(cls, text: str, page_num: int) -> List[Dict[str, Any]]:
        items = []
        for pattern in cls._sec_item_patterns():
            for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                groups = [g for g in m.groups() if g is not None]
                if not groups:
                    continue
                item_num, item_title = None, ""
                if len(groups) == 1:
                    pot = groups[0].strip()
                    if re.match(r"^[\dA-Z]+(\.\d+)?[A-Z]?$", pot):
                        item_num = pot
                elif len(groups) == 2:
                    a, b = groups
                    if re.match(r"^[\dA-Z]+(\.\d+)?[A-Z]?$", a.strip()):
                        item_num = a.strip()
                        item_title = b.strip()[:100]
                    elif re.match(r"^[\dA-Z]+(\.\d+)?[A-Z]?$", b.strip()):
                        if re.match(r"^(Part\s+)?[IVX]+$", a.strip(), re.IGNORECASE):
                            item_num = f"{a.strip()}-{b.strip()}"
                        else:
                            item_num = b.strip()
                elif len(groups) == 3:
                    part, pot, title = groups
                    if re.match(r"^[\dA-Z]+(\.\d+)?[A-Z]?$", pot.strip()):
                        item_num = f"{part.strip()}-{pot.strip()}"
                        item_title = (title or "").strip()[:100]
                if item_num:
                    items.append(
                        {
                            "item_number": item_num,
                            "item_title": item_title,
                            "page": page_num,
                            "position": m.start(),
                        }
                    )
        seen, unique = set(), []
        for it in sorted(items, key=lambda x: x["position"]):
            key = (it["item_number"], it["page"])
            if key not in seen:
                seen.add(key)
                unique.append(it)
        return unique

    @staticmethod
    def _assign_chunk_ids(split_docs: List[Document]) -> None:
        counters: Dict[Tuple[str, int], int] = {}
        for d in split_docs:
            md = d.metadata or {}
            fname = md.get("document_name") or op.basename(md.get("source", ""))
            page = int(md.get("page", 0))
            key = (fname, page)
            seq = counters.get(key, 0)
            d.metadata["chunk_id"] = f"{fname}|p{page}|c{seq}"
            counters[key] = seq + 1

    def _collection_exists(self) -> bool:
        try:
            temp_emb = Embeddingclass(load_model=False)
            vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=temp_emb,
                persist_directory=self.persist_dir,
            )
            hits = vs.similarity_search("test", k=1)
            return len(hits) > 0
        except Exception:
            return False

    def _load_pdfs(self, paths: List[str]) -> List[Document]:
        all_docs: List[Document] = []
        for p in paths:
            loader = PyPDFLoader(p)
            page_docs = loader.load()
            fname = op.basename(p)
            page_docs.sort(key=lambda d: (d.metadata or {}).get("page", 0))
            current_items, last_item_page = [], 0
            for doc in page_docs:
                page_num = doc.metadata.get("page", 0)
                text = doc.page_content
                page_items = Chromaclass._extract_sec_items(text, page_num)
                if page_items:
                    current_items.extend(page_items)
                    current_items = current_items[-5:]
                    last_item_page = page_num
                primary_item = (
                    page_items[0]
                    if page_items
                    else (
                        current_items[-1]
                        if current_items and (page_num - last_item_page <= 5)
                        else None
                    )
                )
                md = {
                    "source": doc.metadata.get("source", p),
                    "page": page_num,
                    "document_name": fname,
                    "has_sec_items": len(page_items) > 0,
                    "item_number": primary_item["item_number"] if primary_item else "",
                    "item_title": primary_item["item_title"] if primary_item else "",
                }
                all_docs.append(Document(page_content=doc.page_content, metadata=md))
        return all_docs

    def _build_index(self):
        pdfs = [p for p in glob.glob(op.join(self.pdf_dir, "*.pdf")) if op.isfile(p)]
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found in {self.pdf_dir}")
        emb = Embeddingclass()
        docs = self._load_pdfs(pdfs)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        split_docs = splitter.split_documents(docs)
        Chromaclass._assign_chunk_ids(split_docs)
        vs = Chroma.from_documents(
            documents=split_docs,
            embedding=emb,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        vs.persist()
        Chromaclass._vectorstore = vs

    def _ensure_vectorstore(self):
        if self._collection_exists():
            Chromaclass._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=Embeddingclass(load_model=False),
                persist_directory=self.persist_dir,
            )
        else:
            self._build_index()

    def vectorstore(self) -> Chroma:
        return Chromaclass._vectorstore

    def retrieve(
        self, query: str, k: int = 10, prefetch: int = 30
    ) -> List[Dict[str, Any]]:
        candidates = self.vectorstore().similarity_search(query, k=prefetch)
        rr = rerankerclass()
        top_docs = rr.rerank(query, candidates, top_k=k) if candidates else []
        results = []
        for doc in top_docs:
            md = doc.metadata or {}
            results.append(
                {
                    "chunk_id": md.get("chunk_id", ""),
                    "content": doc.page_content,
                    "document_name": md.get("document_name", ""),
                    "page_number": int(md.get("page", 0)),
                    "item_number": md.get("item_number", ""),
                    "item_title": md.get("item_title", ""),
                    "rerank_score": float(md.get("rerank_score", 0.0)),
                }
            )
        return results

    def retrieve_enhanced(
        self, query: str, k: int = 10, prefetch: int = 25
    ) -> List[Dict[str, Any]]:
        """Enhanced retrieval with stopword removal and keyword matching"""

        candidates = self.vectorstore().similarity_search(query, k=prefetch * 2)

        enhanced_reranker = Reranker("ms-marco-MiniLM-L-12-v2")

        top_docs = enhanced_reranker.rerank(
            query, candidates, top_k=k, keyword_filter_threshold=0.05, hybrid_weight=0.3
        )

        results = []
        for doc in top_docs:
            md = doc.metadata or {}
            results.append(
                {
                    "chunk_id": md.get("chunk_id", ""),
                    "content": doc.page_content,
                    "document_name": md.get("document_name", ""),
                    "page_number": int(md.get("page", 0)),
                    "item_number": md.get("item_number", ""),
                    "item_title": md.get("item_title", ""),
                    "rerank_score": float(md.get("rerank_score", 0.0)),
                    "keyword_score": float(md.get("keyword_score", 0.0)),
                    "flashrank_score": float(md.get("flashrank_score", 0.0)),
                }
            )
        return results


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
