import { db } from "@/db";
import { openai } from "@/lib/openai";
import { getPineconeClient } from "@/lib/pinecone";
import { SendMessageValidator } from "@/lib/validators/SendMessageValidator";
import { getKindeServerSession } from "@kinde-oss/kinde-auth-nextjs/server";
import { getEmbeddings } from "@/lib/embeddings";
import { PineconeStore } from "@langchain/pinecone";
import { NextRequest } from "next/server";
import { aiServiceEnabled, chatStream } from "@/lib/ai-service";

export const POST = async (req: NextRequest) => {
  // endpoint for asking a question to a pdf file

  const body = await req.json();

  const { getUser } = getKindeServerSession();
  const user = await getUser();

  const userId = user?.id;

  if (!userId) return new Response("Unauthorized", { status: 401 });

  const { fileId, message } = SendMessageValidator.parse(body);

  const file = await db.file.findFirst({
    where: {
      id: fileId,
      userId,
    },
  });

  if (!file) return new Response("Not found", { status: 404 });

  await db.message.create({
    data: {
      text: message,
      isUserMessage: true,
      userId,
      fileId,
    },
  });

  // Agentic RAG service (LangGraph): retrieval, grading, tools, grounding check
  // and page citations happen there. Conversation memory is kept per file.
  if (aiServiceEnabled()) {
    try {
      const stream = await chatStream({
        userId,
        fileId,
        message,
        signal: req.signal,
        onDone: async (answer) => {
          await db.message.create({
            data: { text: answer, isUserMessage: false, fileId, userId },
          });
        },
      });
      return new Response(stream, {
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      });
    } catch (err) {
      console.error("ai-service chat failed", err);
      return new Response("The assistant is unavailable. Try again.", {
        status: 502,
      });
    }
  }

  // Original pipeline (Pinecone + one LLM call), used when AI_SERVICE_URL is unset.
  // 1: vectorize message
  const embeddings = getEmbeddings();

  const pinecone = await getPineconeClient();
  const pineconeIndex = pinecone.Index("blistro");

  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex,
    namespace: file.id,
  });

  const results = await vectorStore.similaritySearch(message, 4);

  const prevMessages = await db.message.findMany({
    where: {
      fileId,
    },
    orderBy: {
      createdAt: "asc",
    },
    take: 6,
  });

  const formattedPrevMessages = prevMessages.map((msg) => ({
    role: msg.isUserMessage ? ("user" as const) : ("assistant" as const),
    content: msg.text,
  }));

  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    temperature: 0,
    stream: true,
    messages: [
      {
        role: "system",
        content:
          "Use the following pieces of context (or previous conversaton if needed) to answer the users question in markdown format.",
      },
      {
        role: "user",
        content: `Use the following pieces of context (or previous conversaton if needed) to answer the users question in markdown format. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.
        
  \n----------------\n
  
  PREVIOUS CONVERSATION:
  ${formattedPrevMessages.map((message) => {
    if (message.role === "user") return `User: ${message.content}\n`;
    return `Assistant: ${message.content}\n`;
  })}
  
  \n----------------\n
  
  CONTEXT:
  ${results.map((r) => r.pageContent).join("\n\n")}
  
  USER INPUT: ${message}`,
      },
    ],
  });

  // Stream the answer as plain text (same wire format the old `ai`
  // StreamingTextResponse used), then save the full answer.
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let completion = "";
      try {
        for await (const chunk of response) {
          const text = chunk.choices[0]?.delta?.content ?? "";
          if (!text) continue;
          completion += text;
          controller.enqueue(encoder.encode(text));
        }

        await db.message.create({
          data: {
            text: completion,
            isUserMessage: false,
            fileId,
            userId,
          },
        });

        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
};
