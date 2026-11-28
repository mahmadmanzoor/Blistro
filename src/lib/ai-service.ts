// Client for the Python agentic RAG service (ai-service/). Turned on by setting
// AI_SERVICE_URL and AI_SERVICE_TOKEN; without them Blistro keeps its original
// Pinecone pipeline. Server-only: the token must never reach the browser.

const baseUrl = process.env.AI_SERVICE_URL?.replace(/\/$/, "");
const token = process.env.AI_SERVICE_TOKEN;

export const aiServiceEnabled = () => Boolean(baseUrl && token);

const headers = (userId: string) => ({
  "Content-Type": "application/json",
  "X-Service-Token": token!,
  "X-User-Id": userId,
});

export async function registerDocument(opts: {
  userId: string;
  documentId: string;
  name: string;
  url: string;
}): Promise<{ job_id: string }> {
  const res = await fetch(`${baseUrl}/documents`, {
    method: "POST",
    headers: headers(opts.userId),
    body: JSON.stringify({
      document_id: opts.documentId,
      name: opts.name,
      url: opts.url,
    }),
  });
  if (!res.ok) throw new Error(`ai-service /documents ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function getDocumentStatus(
  userId: string,
  documentId: string
): Promise<"pending" | "processing" | "ready" | "failed" | null> {
  const res = await fetch(`${baseUrl}/documents/${encodeURIComponent(documentId)}`, {
    headers: headers(userId),
    cache: "no-store",
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`ai-service /documents/${documentId} ${res.status}`);
  return (await res.json()).status;
}

type ChatEvent =
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: { answer: string; ttft_ms: number | null; total_ms: number } }
  | { event: "error"; data: { message: string } }
  | { event: string; data: Record<string, unknown> };

async function* readSse(body: ReadableStream<Uint8Array>): AsyncGenerator<ChatEvent> {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let sep: number;
    while ((sep = buffer.search(/\r?\n\r?\n/)) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(buffer[sep] === "\r" ? sep + 4 : sep + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of raw.split(/\r?\n/)) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue;
      try {
        yield { event, data: JSON.parse(data.join("\n")) } as ChatEvent;
      } catch {
        // ignore malformed frames (keep-alive comments etc.)
      }
    }
  }
}

/**
 * Ask the agent about one file. Returns a plain-text stream (the wire format
 * Blistro's chat UI already reads) and calls onDone with the final answer, in
 * which [S1]-style labels are already replaced by page citations.
 */
export async function chatStream(opts: {
  userId: string;
  fileId: string;
  message: string;
  signal?: AbortSignal;
  onDone: (answer: string) => Promise<void>;
}): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch(`${baseUrl}/chat`, {
    method: "POST",
    headers: { ...headers(opts.userId), Accept: "text/event-stream" },
    body: JSON.stringify({
      document_ids: [opts.fileId],
      message: opts.message,
      thread_id: opts.fileId,
    }),
    signal: opts.signal,
  });
  if (!res.ok || !res.body) throw new Error(`ai-service /chat ${res.status}`);

  const encoder = new TextEncoder();
  const events = readSse(res.body);
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        for await (const ev of events) {
          if (ev.event === "token") {
            controller.enqueue(encoder.encode((ev.data as { text: string }).text));
          } else if (ev.event === "reset") {
            // The agent regenerated after a failed grounding check. A plain-text
            // stream can't retract text, so mark the restart; the saved message
            // (shown after the stream ends) holds only the final answer.
            controller.enqueue(encoder.encode("\n\n_Checking sources again..._\n\n"));
          } else if (ev.event === "done") {
            await opts.onDone((ev.data as { answer: string }).answer);
          } else if (ev.event === "error") {
            throw new Error((ev.data as { message: string }).message);
          }
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
    async cancel() {
      await events.return(undefined);
    },
  });
}
