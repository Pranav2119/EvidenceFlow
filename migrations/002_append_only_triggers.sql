-- 002_append_only_triggers.sql: Enforces append-only immutability on audit tables.

-- Row-level trigger function: aborts any UPDATE or DELETE
CREATE OR REPLACE FUNCTION enforce_append_only()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Security Violation: Table "%" is append-only. UPDATE and DELETE operations are strictly forbidden.', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

-- Statement-level trigger function: aborts any TRUNCATE
CREATE OR REPLACE FUNCTION enforce_no_truncate()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Security Violation: Table "%" is append-only. TRUNCATE is strictly forbidden.', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

-- Apply row-level UPDATE/DELETE triggers
DROP TRIGGER IF EXISTS trg_step_ledger_append_only ON step_ledger;
CREATE TRIGGER trg_step_ledger_append_only
BEFORE UPDATE OR DELETE ON step_ledger
FOR EACH ROW EXECUTE FUNCTION enforce_append_only();

DROP TRIGGER IF EXISTS trg_claim_audit_append_only ON claim_audit;
CREATE TRIGGER trg_claim_audit_append_only
BEFORE UPDATE OR DELETE ON claim_audit
FOR EACH ROW EXECUTE FUNCTION enforce_append_only();

DROP TRIGGER IF EXISTS trg_turns_append_only ON turns;
CREATE TRIGGER trg_turns_append_only
BEFORE UPDATE OR DELETE ON turns
FOR EACH ROW EXECUTE FUNCTION enforce_append_only();

-- Apply statement-level TRUNCATE triggers
DROP TRIGGER IF EXISTS trg_step_ledger_no_truncate ON step_ledger;
CREATE TRIGGER trg_step_ledger_no_truncate
BEFORE TRUNCATE ON step_ledger
FOR EACH STATEMENT EXECUTE FUNCTION enforce_no_truncate();

DROP TRIGGER IF EXISTS trg_claim_audit_no_truncate ON claim_audit;
CREATE TRIGGER trg_claim_audit_no_truncate
BEFORE TRUNCATE ON claim_audit
FOR EACH STATEMENT EXECUTE FUNCTION enforce_no_truncate();

DROP TRIGGER IF EXISTS trg_turns_no_truncate ON turns;
CREATE TRIGGER trg_turns_no_truncate
BEFORE TRUNCATE ON turns
FOR EACH STATEMENT EXECUTE FUNCTION enforce_no_truncate();

-- Revoke mutation permissions from public/app users (grant SELECT and INSERT only)
REVOKE UPDATE, DELETE, TRUNCATE ON step_ledger FROM PUBLIC;
REVOKE UPDATE, DELETE, TRUNCATE ON claim_audit FROM PUBLIC;
REVOKE UPDATE, DELETE, TRUNCATE ON turns FROM PUBLIC;
