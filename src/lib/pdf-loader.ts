import { Document } from "@langchain/core/documents";

// Replacement for `PDFLoader` from the old `langchain` package (removed because
// it pulled in the unpatchable `expr-eval`). Same approach as that loader:
// the pdf.js build bundled with `pdf-parse`, one Document per page.
export async function loadPdfPages(blob: Blob): Promise<Document[]> {
  // @ts-expect-error -- pdf-parse ships no types for its bundled pdf.js build
  const pdfjs = await import("pdf-parse/lib/pdf.js/v1.10.100/build/pdf.js");
  const { getDocument, version } = pdfjs.default ?? pdfjs;

  const data = new Uint8Array(await blob.arrayBuffer());
  const pdf = await getDocument({
    data,
    useWorkerFetch: false,
    isEvalSupported: false,
    useSystemFonts: true,
  }).promise;
  const meta = await pdf.getMetadata().catch(() => null);

  const documents: Document[] = [];
  for (let i = 1; i <= pdf.numPages; i += 1) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    if (content.items.length === 0) continue;

    const text = content.items
      .map((item: { str: string }) => item.str)
      .join("\n");

    documents.push(
      new Document({
        pageContent: text,
        metadata: {
          source: "blob",
          blobType: blob.type,
          pdf: {
            version,
            info: meta?.info,
            metadata: meta?.metadata,
            totalPages: pdf.numPages,
          },
          loc: { pageNumber: i },
        },
      })
    );
  }

  return documents;
}
