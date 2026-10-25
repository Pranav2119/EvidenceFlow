-- 003_app_role.sql: Creates limited runtime application role without UPDATE/DELETE/TRUNCATE permissions.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'evidenceflow_app') THEN
        CREATE ROLE evidenceflow_app WITH LOGIN PASSWORD 'evidenceflow_app_secret';
    END IF;
END
$$;

-- Grant database connect & schema usage
GRANT CONNECT ON DATABASE evidenceflow_db TO evidenceflow_app;
GRANT USAGE ON SCHEMA public TO evidenceflow_app;

-- Grant SELECT and INSERT ONLY on immutable audit tables
GRANT SELECT, INSERT ON turns, step_ledger, claim_audit TO evidenceflow_app;
REVOKE UPDATE, DELETE, TRUNCATE ON turns, step_ledger, claim_audit FROM evidenceflow_app;

-- Grant necessary privileges on LangGraph checkpointer tables (LangGraph requires upsert/update for checkpoints)
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'checkpoints') THEN
        GRANT SELECT, INSERT, UPDATE ON checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations TO evidenceflow_app;
    END IF;
END
$$;

-- Grant usage on sequences
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO evidenceflow_app;

-- Default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT ON TABLES TO evidenceflow_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE UPDATE, DELETE, TRUNCATE ON TABLES FROM evidenceflow_app;
