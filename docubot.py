"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

from matplotlib import text

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.build_index()

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)

                # split into chunks
                chunks = self.split_into_chunks(text)
                for chunk in chunks:
                    chunk = chunk.strip()
                    if chunk:  # ignore empty chunks
                        docs.append((filename, chunk))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self):
        self.index = {}
        for doc_id, (filename, text) in enumerate(self.documents):
            words = text.lower().split()
            for word in words:
                if word not in self.index:
                    self.index[word] = set()
                self.index[word].add(doc_id)

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def split_into_chunks(self, text):
        return text.split("\n\n")

    def score_document(self, query, doc_tuple):
        """
        doc_tuple = (filename, text)
        Returns a numeric relevance score.
        """
        _, doc_text = doc_tuple
        score = 0
        query_words = query.lower().split()
        doc_words = doc_text.lower().split()

        for word in query_words:
            score += doc_words.count(word)

        return score

    def retrieve(self, query, top_k=3):
        scores = []

        for doc_tuple in self.documents:
            score = self.score_document(query, doc_tuple)
            scores.append((score, doc_tuple))

        scores.sort(reverse=True, key=lambda x: x[0])

        # return top_k snippets only if score > 0
        results = [doc for score, doc in scores if score > 0][:top_k]

        if not results:
            return []

        return results

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
