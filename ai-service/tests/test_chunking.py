from app.embeddings import FakeEmbedder
from app.ingest.chunking import chunk_document, chunk_fixed, chunk_recursive, count_tokens
from app.ingest.pdf import parse_pdf


def test_parse_detects_headings_and_pages(sample_pdf):
    doc = parse_pdf(sample_pdf)
    assert doc.page_count == 3
    headings = [b.text for b in doc.blocks if b.is_heading]
    assert "2 Inspection Schedule" in headings
    assert {b.page for b in doc.blocks} == {1, 2, 3}


def test_fixed_respects_size_and_overlap(sample_pdf):
    doc = parse_pdf(sample_pdf)
    chunks = chunk_fixed(doc, size=60, overlap=10)
    assert len(chunks) > 3
    assert all(c.token_count <= 60 for c in chunks)
    assert all(1 <= c.page_start <= c.page_end <= 3 for c in chunks)


def test_recursive_keeps_headings_and_page_ranges(sample_pdf):
    doc = parse_pdf(sample_pdf)
    chunks = chunk_recursive(doc, size=80)
    assert all(count_tokens(c.content) <= 80 for c in chunks)
    schedule = [c for c in chunks if c.heading == "2 Inspection Schedule"]
    assert schedule and all(c.page_start == 2 for c in schedule)
    assert schedule[0].embed_text.startswith("2 Inspection Schedule\n")


async def test_semantic_and_dispatch(sample_pdf):
    doc = parse_pdf(sample_pdf)
    emb = FakeEmbedder(64)
    for strategy in ("fixed", "recursive", "semantic"):
        chunks = await chunk_document(doc, strategy, emb, size=120, overlap=16)
        assert chunks, strategy
        text = " ".join(c.content for c in chunks)
        assert "XR-200" in text and "safety officer" in text
