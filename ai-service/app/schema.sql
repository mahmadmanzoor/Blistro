-- Blistro AI service schema. Idempotent: safe to run on every deploy.
create extension if not exists vector;

create table if not exists rag_documents (
  id          text primary key,            -- Blistro File.id (cuid)
  user_id     text not null,               -- Kinde user id (= Blistro User.id)
  name        text not null,
  source_url  text,
  page_count  int,
  status      text not null default 'pending',  -- pending | processing | ready | failed
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);
create index if not exists rag_documents_user on rag_documents (user_id, created_at desc);

create table if not exists rag_chunks (
  id           bigserial primary key,
  document_id  text not null references rag_documents(id) on delete cascade,
  user_id      text not null,
  strategy     text not null,              -- fixed | recursive | semantic
  chunk_index  int  not null,
  page_start   int  not null,
  page_end     int  not null,
  heading      text,
  content      text not null,
  token_count  int  not null,
  embedding    vector(1536) not null,
  tsv          tsvector generated always as (
                 to_tsvector('english', coalesce(heading, '') || ' ' || content)
               ) stored,
  unique (document_id, strategy, chunk_index)
);
-- ANN index for cosine distance. Filters (user, strategy, document) are applied
-- with hnsw.iterative_scan so a selective filter still returns k rows.
create index if not exists rag_chunks_embedding_hnsw
  on rag_chunks using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);
create index if not exists rag_chunks_tsv on rag_chunks using gin (tsv);
create index if not exists rag_chunks_scope on rag_chunks (user_id, strategy, document_id);
create index if not exists rag_chunks_page on rag_chunks (document_id, strategy, page_start);

create table if not exists rag_jobs (
  id           uuid primary key,
  document_id  text not null references rag_documents(id) on delete cascade,
  user_id      text not null,
  status       text not null default 'queued',  -- queued | running | done | failed
  attempts     int  not null default 0,
  error        text,
  progress     jsonb not null default '{}'::jsonb,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  locked_at    timestamptz
);
-- Workers claim with FOR UPDATE SKIP LOCKED; this partial index keeps that scan tiny.
create index if not exists rag_jobs_queued on rag_jobs (created_at) where status = 'queued';
create index if not exists rag_jobs_document on rag_jobs (document_id);
