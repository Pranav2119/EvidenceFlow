-- 004_corpus_chunks.sql: Corpus chunks storage for live deterministic hash validation.

CREATE TABLE IF NOT EXISTS corpus_chunks (
    chunk_id VARCHAR(128) PRIMARY KEY,
    version VARCHAR(16) NOT NULL,
    doc_name VARCHAR(256),
    section TEXT,
    title TEXT,
    content TEXT NOT NULL,
    sha256 VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_corpus_chunks_version ON corpus_chunks(version);
CREATE INDEX IF NOT EXISTS idx_corpus_chunks_sha256 ON corpus_chunks(sha256);

-- Grant permissions to evidenceflow_app role
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'evidenceflow_app') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE ON corpus_chunks TO evidenceflow_app;
    END IF;
END
$$;
