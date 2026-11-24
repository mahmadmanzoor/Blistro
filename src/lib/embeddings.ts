import { OpenAIEmbeddings } from "@langchain/openai";

// Pinned to the model the old `langchain` package used by default, so new
// query vectors stay compatible with the vectors already stored in Pinecone.
export const getEmbeddings = () =>
  new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: "text-embedding-ada-002",
  });
