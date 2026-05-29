from __future__ import annotations

from pathlib import Path
import json
import sys

import streamlit as st

try:
    from .answer import answer_from_hits
    from .embeddings import DEFAULT_MODEL_NAME
    from .eval import load_eval_set, run_eval
    from .index import LocalFaissIndex
    from .ingest import ingest_pdfs
    from .retrieve import Retriever
    from .storage import LocalStore
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from app.answer import answer_from_hits
    from app.embeddings import DEFAULT_MODEL_NAME
    from app.eval import load_eval_set, run_eval
    from app.index import LocalFaissIndex
    from app.ingest import ingest_pdfs
    from app.retrieve import Retriever
    from app.storage import LocalStore


ROOT = Path(__file__).resolve().parents[1]


def _get_runtime():
    store = LocalStore(ROOT)
    faiss_index = LocalFaissIndex(store)
    faiss_index.load()
    return store, faiss_index


def render_app() -> None:
    st.set_page_config(page_title="Local PDF QA", layout="wide")
    st.title("Local-first PDF Question Answering")
    st.caption("Upload PDFs, index them locally, and answer questions with page citations only from retrieved chunks.")

    store, faiss_index = _get_runtime()
    model_name = st.sidebar.text_input("Embedding model", value=DEFAULT_MODEL_NAME)
    use_keyword_search = st.sidebar.checkbox("Use keyword search rerank", value=True)
    top_k = st.sidebar.slider("Top-k retrieval", 1, 10, 5)

    uploaded = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        if st.button("Ingest uploaded PDFs"):
            temp_dir = ROOT / "data" / "uploads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for file in uploaded:
                temp_path = temp_dir / f"_tmp_{file.name}"
                temp_path.write_bytes(file.getbuffer())
                paths.append(temp_path)
            with st.spinner("Indexing documents locally..."):
                results = ingest_pdfs(paths, store=store, faiss_index=faiss_index, model_name=model_name)
            for r in results:
                st.success(f"Ingested {r.document.file_name}: {r.pages_processed} pages, {r.chunks_added} chunks")
            st.rerun()

    st.divider()
    st.subheader("Ask a question")
    question = st.text_input("Question", placeholder="Ask something grounded in the uploaded PDFs")
    if st.button("Answer question", disabled=not question.strip()):
        retriever = Retriever(store=store, faiss_index=faiss_index, model_name=model_name)
        retrieval = retriever.retrieve(question, top_k=top_k, use_keyword_search=use_keyword_search)
        answer = answer_from_hits(question, retrieval.hits)

        st.markdown("### Answer")
        st.write(answer.answer)

        st.markdown("### Citations")
        if answer.citations:
            for citation in answer.citations:
                st.write(citation)
        else:
            st.write("No supporting citations found.")

        with st.expander("Retrieved chunks"):
            for hit in retrieval.hits:
                st.write(
                    {
                        "citation": hit.chunk.citation(),
                        "semantic_score": round(hit.semantic_score, 4),
                        "keyword_score": round(hit.keyword_score, 4),
                        "combined_score": round(hit.combined_score, 4),
                        "source_type": hit.chunk.source_type,
                        "text": hit.chunk.chunk_text,
                    }
                )

    st.divider()
    st.subheader("Evaluation mode")
    eval_path = st.text_input("Evaluation JSON path", value=str(ROOT / "data" / "eval" / "sample_eval.json"))
    demo_pdf = ROOT / "data" / "eval" / "demo_policy.pdf"
    if st.button("Load demo evaluation PDF") and demo_pdf.exists():
        with st.spinner("Indexing the bundled demo PDF..."):
            ingest_pdfs([demo_pdf], store=store, faiss_index=faiss_index, model_name=model_name)
        st.success("Demo PDF indexed.")
        st.rerun()

    if st.button("Run evaluation"):
        retriever = Retriever(store=store, faiss_index=faiss_index, model_name=model_name)
        try:
            eval_cases = load_eval_set(eval_path)
        except Exception as exc:
            st.error(f"Could not load eval file: {exc}")
        else:
            results = run_eval(retriever, eval_cases)
            st.write([
                {
                    "question": r.question,
                    "answer": r.answer,
                    "citations": r.citations,
                    "hit": r.hit,
                    "refused": r.refused,
                }
                for r in results
            ])


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
