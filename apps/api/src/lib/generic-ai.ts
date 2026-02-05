import { createOpenAI } from "@ai-sdk/openai";
import { config } from "../config";
import { createOllama } from "ollama-ai-provider";
import { anthropic } from "@ai-sdk/anthropic";
import { groq } from "@ai-sdk/groq";
import { google } from "@ai-sdk/google";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { fireworks } from "@ai-sdk/fireworks";
import { createVertex } from "@ai-sdk/google-vertex";

type Provider =
  | "openai"
  | "ollama"
  | "anthropic"
  | "groq"
  | "google"
  | "openrouter"
  | "fireworks"
  | "deepinfra"
  | "vertex";

// For DeepInfra, use OpenAI-compatible API with chat endpoint
const deepinfraOpenAI = createOpenAI({
  apiKey: config.DEEPINFRA_API_KEY,
  baseURL: "https://api.deepinfra.com/v1/openai",
  name: "deepinfra",
});

const defaultProvider: Provider = config.OLLAMA_BASE_URL
  ? "ollama"
  : config.DEEPINFRA_API_KEY
    ? "deepinfra"
    : "openai";

export function getDefaultProvider(): Provider {
  return defaultProvider;
}

const openaiProvider = createOpenAI({
  apiKey: config.OPENAI_API_KEY,
  baseURL: config.OPENAI_BASE_URL,
});

const providerList: Record<Provider, any> = {
  openai: openaiProvider,
  ollama: createOllama({
    baseURL: config.OLLAMA_BASE_URL,
  }),
  anthropic, //ANTHROPIC_API_KEY
  groq, //GROQ_API_KEY
  google, //GOOGLE_GENERATIVE_AI_API_KEY
  openrouter: createOpenRouter({
    apiKey: config.OPENROUTER_API_KEY,
  }),
  fireworks, //FIREWORKS_API_KEY
  deepinfra: deepinfraOpenAI,
  vertex: createVertex({
    project: "firecrawl",
    //https://github.com/vercel/ai/issues/6644 bug
    baseURL:
      "https://aiplatform.googleapis.com/v1/projects/firecrawl/locations/global/publishers/google",
    location: "global",
    googleAuthOptions: config.VERTEX_CREDENTIALS
      ? {
          credentials: JSON.parse(atob(config.VERTEX_CREDENTIALS)),
        }
      : {
          keyFile: "./gke-key.json",
        },
  }),
};

// Providers that need .chat() to use Chat Completions API instead of Responses API
const chatProviders = new Set(["openai", "deepinfra"]);

export function getModel(name: string, provider: Provider = defaultProvider) {
  if (name === "gemini-2.5-pro") {
    name = "gemini-2.5-pro";
  }

  const modelName = config.MODEL_NAME || name;
  const providerInstance = providerList[provider];

  // Use .chat() for OpenAI-compatible providers to force Chat Completions API
  if (chatProviders.has(provider) && providerInstance.chat) {
    return providerInstance.chat(modelName);
  }

  return providerInstance(modelName);
}

export function getEmbeddingModel(
  name: string,
  provider: Provider = defaultProvider,
) {
  return config.MODEL_EMBEDDING_NAME
    ? providerList[provider].embedding(config.MODEL_EMBEDDING_NAME)
    : providerList[provider].embedding(name);
}
