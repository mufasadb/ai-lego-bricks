-- ==============================================================================
-- Supabase pgvector Setup for Memory Service
-- ==============================================================================
-- This script sets up pgvector extension and creates the necessary tables and functions
-- Run this in your Supabase SQL Editor

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Create memories table with vector support
create table if not exists memories (
    id uuid primary key default gen_random_uuid(),
    content text not null,
    embedding vector(384), -- 384 dimensions for all-MiniLM-L6-v2 model
    metadata jsonb default '{}',
    created_at timestamp with time zone default now(),
    updated_at timestamp with time zone default now()
);

-- 3. Create index for vector similarity search
create index if not exists memories_embedding_idx 
on memories using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- 4. Create index for metadata queries
create index if not exists memories_metadata_idx 
on memories using gin (metadata);

-- 5. Create index for text search
create index if not exists memories_content_idx 
on memories using gin (to_tsvector('english', content));

-- 6. Create function for vector similarity search
create or replace function match_memories(
    query_embedding vector(384),
    match_threshold float default 0.5,
    match_count int default 10
)
returns table (
    id uuid,
    content text,
    metadata jsonb,
    created_at timestamp with time zone,
    updated_at timestamp with time zone,
    similarity float
)
language sql stable
as $$
    select
        memories.id,
        memories.content,
        memories.metadata,
        memories.created_at,
        memories.updated_at,
        (memories.embedding <=> query_embedding) * -1 + 1 as similarity
    from memories
    where memories.embedding <=> query_embedding < 1 - match_threshold
    order by memories.embedding <=> query_embedding
    limit match_count;
$$;

-- 7. Create function for hybrid search (vector + text)
create or replace function hybrid_search_memories(
    query_text text,
    query_embedding vector(384),
    match_threshold float default 0.5,
    match_count int default 10
)
returns table (
    id uuid,
    content text,
    metadata jsonb,
    created_at timestamp with time zone,
    updated_at timestamp with time zone,
    similarity float,
    text_rank float
)
language sql stable
as $$
    with vector_search as (
        select
            memories.id,
            memories.content,
            memories.metadata,
            memories.created_at,
            memories.updated_at,
            (memories.embedding <=> query_embedding) * -1 + 1 as similarity
        from memories
        where memories.embedding <=> query_embedding < 1 - match_threshold
        order by memories.embedding <=> query_embedding
        limit match_count * 2
    ),
    text_search as (
        select
            memories.id,
            ts_rank_cd(to_tsvector('english', memories.content), plainto_tsquery('english', query_text)) as text_rank
        from memories
        where to_tsvector('english', memories.content) @@ plainto_tsquery('english', query_text)
    )
    select
        vector_search.id,
        vector_search.content,
        vector_search.metadata,
        vector_search.created_at,
        vector_search.updated_at,
        vector_search.similarity,
        coalesce(text_search.text_rank, 0) as text_rank
    from vector_search
    left join text_search on vector_search.id = text_search.id
    order by 
        (vector_search.similarity * 0.7 + coalesce(text_search.text_rank, 0) * 0.3) desc
    limit match_count;
$$;

-- 8. Create trigger to automatically update the updated_at timestamp
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

create trigger update_memories_updated_at
    before update on memories
    for each row
    execute function update_updated_at_column();

-- 9. Enable Row Level Security (RLS) for the memories table
alter table memories enable row level security;

-- 10. Create policy to allow all operations for authenticated users
-- Note: Adjust this policy based on your security requirements
create policy "Allow all operations for authenticated users" on memories
    for all using (auth.role() = 'authenticated' or auth.role() = 'anon');

-- ==============================================================================
-- Verification Queries
-- ==============================================================================
-- Run these to verify the setup worked correctly:

-- Check if pgvector extension is installed
-- SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check table structure
-- \d memories;

-- Check indexes
-- SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'memories';

-- Check functions
-- SELECT routine_name, routine_type FROM information_schema.routines 
-- WHERE routine_schema = 'public' AND routine_name LIKE '%memor%';

-- ==============================================================================
-- Sample Usage
-- ==============================================================================
-- Insert a test memory:
-- INSERT INTO memories (content, embedding, metadata) 
-- VALUES (
--     'This is a test memory about machine learning',
--     '[0.1, 0.2, 0.3, ...]'::vector,  -- Replace with actual 384-dim vector
--     '{"type": "test", "category": "ml"}'::jsonb
-- );

-- Search for similar memories:
-- SELECT * FROM match_memories(
--     '[0.1, 0.2, 0.3, ...]'::vector,  -- Query vector
--     0.5,  -- Similarity threshold
--     5     -- Number of results
-- );