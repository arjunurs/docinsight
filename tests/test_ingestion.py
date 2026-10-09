"""
Ingestion tests: single chunking pass, env config, idempotent re-uploads, reload after restart.

Embeddings are mocked, so no OpenAI calls are made.
"""
import pytest
from llama_index.core.embeddings import MockEmbedding

import utils.index_manager as index_manager_module
from utils.config import AppConfig
from utils.document_processor import DocumentProcessor
from utils.index_manager import IndexManager

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(400))


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        index_manager_module, "OpenAIEmbedding", lambda model: MockEmbedding(embed_dim=8)
    )


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "upload_a" / "notes.txt"
    path.parent.mkdir()
    path.write_text(TEXT)
    return path


def make_manager(tmp_path):
    return IndexManager(persist_dir=str(tmp_path / "chroma"), collection_name="test")


def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("CHUNK_OVERLAP", "25")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "some-embedding")
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)

    config = AppConfig.from_env()

    assert (config.chunk_size, config.chunk_overlap) == (256, 25)
    assert config.llm_model == "some-model"
    assert config.embedding_model == "some-embedding"
    assert config.llm_temperature == 0.0


def test_default_temperature_is_zero(tmp_path):
    assert make_manager(tmp_path).llm.temperature == 0.0


def test_index_stores_exactly_the_processor_chunks(tmp_path, sample_file):
    processor = DocumentProcessor(chunk_size=128, chunk_overlap=16)
    nodes = processor.process_documents(processor.load_documents([str(sample_file)]))
    manager = make_manager(tmp_path)

    manager.add_nodes(nodes)

    stored = manager.chroma_collection.get()
    assert len(nodes) > 1
    assert sorted(stored["ids"]) == sorted(node.node_id for node in nodes)


def test_reupload_does_not_duplicate(tmp_path, sample_file):
    processor = DocumentProcessor(chunk_size=128, chunk_overlap=16)
    manager = make_manager(tmp_path)
    first = manager.add_nodes(processor.process_documents(processor.load_documents([str(sample_file)])))

    # Same bytes, different temp directory and file name, as a fresh Streamlit upload would be
    copy = tmp_path / "upload_b" / "renamed.txt"
    copy.parent.mkdir()
    copy.write_bytes(sample_file.read_bytes())
    second = manager.add_nodes(processor.process_documents(processor.load_documents([str(copy)])))

    assert len(first) > 0
    assert second == []
    assert manager.chroma_collection.count() == len(first)


def test_index_is_queryable_after_restart(tmp_path, sample_file):
    processor = DocumentProcessor(chunk_size=128, chunk_overlap=16)
    added = make_manager(tmp_path).add_nodes(
        processor.process_documents(processor.load_documents([str(sample_file)]))
    )

    restarted = make_manager(tmp_path)

    assert restarted.has_documents()
    hits = restarted.get_index().as_retriever(similarity_top_k=3).retrieve("topic 3")
    assert len(hits) == 3
    assert {hit.node.node_id for hit in hits} <= {node.node_id for node in added}
