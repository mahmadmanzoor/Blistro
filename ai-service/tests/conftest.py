import os

# Tests never call OpenAI or Langfuse, and never read the developer's .env.
os.environ.update(
    {
        "ENV": "test",
        "AUTH_MODE": "dev",
        "EMBEDDINGS_PROVIDER": "fake",
        "RERANK_PROVIDER": "none",
        "OPENAI_API_KEY": "sk-test-not-used",
        "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SECRET_KEY": "",
        "COHERE_API_KEY": "",
    }
)
os.environ.setdefault(
    "DATABASE_URL", "postgresql://blistro:blistro@localhost:55433/blistro_ai_test"
)

import pymupdf  # noqa: E402
import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()


def make_pdf(pages: list[list[tuple[str, float, bool]]]) -> bytes:
    """Each page is a list of (text, font size, bold) lines."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for text, size, bold in lines:
            rect = pymupdf.Rect(72, y, 540, y + 400)
            used = page.insert_textbox(
                rect, text, fontsize=size, fontname="hebo" if bold else "helv"
            )
            # insert_textbox returns the unused height; advance by what was used.
            y += (400 - used) + size
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def sample_pdf() -> bytes:
    body = (
        "The XR-200 valve must be inspected every 90 days. Inspectors record the "
        "pressure reading and the seal condition in the maintenance log. "
    ) * 4
    return make_pdf(
        [
            [("1 Introduction", 18, True), (body, 10, False)],
            [("2 Inspection Schedule", 18, True), (body, 10, False), (body, 10, False)],
            [("3 Reporting", 18, True), ("Reports go to the safety officer. " * 10, 10, False)],
        ]
    )
