-- 001_initial_schema.sql: Core schema for turns, step_ledger, and claim_audit.

CREATE TABLE IF NOT EXISTS turns (
    turn_id UUID PRIMARY KEY,
    thread_id VARCHAR(128) NOT NULL,
    turn_number INT NOT NULL DEFAULT 1,
    target_version VARCHAR(16) NOT NULL,
    query TEXT NOT NULL,
    draft_answer TEXT NOT NULL,
    is_abstained BOOLEAN NOT NULL DEFAULT FALSE,
    abstention_code VARCHAR(64),
    abstention_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS step_ledger (
    step_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id UUID NOT NULL,
    thread_id VARCHAR(128) NOT NULL,
    step_number INT NOT NULL,
    node VARCHAR(64) NOT NULL,
    action VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL,
    abstention_code VARCHAR(64),
    decided_by VARCHAR(32) NOT NULL,
    latency_ms FLOAT NOT NULL,
    tokens JSONB DEFAULT '{}'::jsonb,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS claim_audit (
    claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    turn_id UUID NOT NULL,
    thread_id VARCHAR(128) NOT NULL,
    statement TEXT NOT NULL,
    chunk_id VARCHAR(128) NOT NULL,
    chunk_sha256 VARCHAR(64) NOT NULL,
    chunk_version VARCHAR(16) NOT NULL,
    claimed_values JSONB DEFAULT '[]'::jsonb,
    verifier_verdict VARCHAR(16) NOT NULL,
    verifier_code VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Fast lookup indexes
CREATE INDEX IF NOT EXISTS idx_turns_thread ON turns(thread_id);
CREATE INDEX IF NOT EXISTS idx_step_ledger_thread_turn ON step_ledger(thread_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_claim_audit_thread_turn ON claim_audit(thread_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_claim_audit_chunk ON claim_audit(chunk_id, chunk_sha256);
