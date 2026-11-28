# Blistro - A Modern Fullstack SaaS-Platform

Built with the Next.js 13.5 App Router, tRPC, TypeScript, Prisma & Tailwind

![Project Image](https://github.com/mahmadmanzoor/Blistro/blob/main/public/thumbnail.png)

## Features

- 🛠️ Complete SaaS Built From Scratch
- 💻 Beautiful Landing Page & Pricing Page Included
- 💳 Free & Pro Plan Using Stripe
- 📄 A Beautiful And Highly Functional PDF Viewer
- 🔄 Streaming API Responses in Real-Time
- 🔒 Authentication Using Kinde
- 🎨 Clean, Modern UI Using 'shadcn-ui'
- 🚀 Optimistic UI Updates for a Great UX
- ⚡ Infinite Message Loading for Performance
- 📤 Intuitive Drag n’ Drop Uploads
- ✨ Instant Loading States
- 🔧 Modern Data Fetching Using tRPC & Zod
- 🧠 LangChain for Infinite AI Memory
- 🌲 Pinecone as our Vector Storage
- 📊 Prisma as our ORM
- 🔤 100% written in TypeScript
- 🎁 ...much more

## Getting started

To get started with this project, run

```bash
  git clone https://github.com/mahmadmanzoor/Blistro.git
```

and copy the .env.example variables into a separate .env file, fill them out & and that's all you need to get started!


## Acknowledgements

- [Kinde Auth](https://kinde.com) for making this project possible

## License

[MIT](https://choosealicense.com/licenses/mit/)
## Agentic RAG service

`ai-service/` is a Python LangGraph agent (FastAPI, Postgres + pgvector, Langfuse)
that replaces the original single-call RAG when `AI_SERVICE_URL` and
`AI_SERVICE_TOKEN` are set. See [ai-service/README.md](ai-service/README.md).
