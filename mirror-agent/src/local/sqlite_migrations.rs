//! Ordered SQLite schema migrations for agent-owned state.

use crate::StoreError;
use rusqlite::Connection;
use rusqlite::TransactionBehavior;

pub(super) mod v3_context_windows;

struct Migration {
    version: i64,
    sql: &'static str,
}

const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        sql: include_str!("../../migrations/0001_initial.sql"),
    },
    Migration {
        version: 2,
        sql: include_str!("../../migrations/0002_session_lineage.sql"),
    },
    Migration {
        version: 3,
        sql: include_str!("../../migrations/0003_system_prompts.sql"),
    },
];

pub(super) fn migrate(connection: &mut Connection) -> Result<(), StoreError> {
    migrate_to(connection, MIGRATIONS)
}

fn migrate_to(connection: &mut Connection, migrations: &[Migration]) -> Result<(), StoreError> {
    validate_definitions(migrations)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(persistence)?;
    let current: i64 = transaction
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(persistence)?;
    if current < 0 {
        return Err(migration_error(format!(
            "agent database schema version {current} is negative"
        )));
    }
    let latest = migrations.last().map_or(0, |migration| migration.version);
    if current > latest {
        return Err(migration_error(format!(
            "agent database schema version {current} is newer than this binary"
        )));
    }
    for migration in migrations
        .iter()
        .filter(|migration| migration.version > current)
    {
        transaction
            .execute_batch(migration.sql)
            .map_err(persistence)?;
        if migration.version == 3 {
            v3_context_windows::migrate(&transaction)?;
        }
        transaction
            .pragma_update(None, "user_version", migration.version)
            .map_err(persistence)?;
    }
    transaction.commit().map_err(persistence)
}

fn validate_definitions(migrations: &[Migration]) -> Result<(), StoreError> {
    for (index, migration) in migrations.iter().enumerate() {
        let expected = i64::try_from(index + 1).map_err(|_| {
            migration_error("agent migration count exceeds SQLite INTEGER".to_string())
        })?;
        if migration.version != expected {
            return Err(migration_error(format!(
                "expected agent migration {expected}, got {}",
                migration.version
            )));
        }
    }
    Ok(())
}

fn migration_error(message: String) -> StoreError {
    StoreError::Persistence(message)
}

fn persistence(error: impl std::fmt::Display) -> StoreError {
    StoreError::Persistence(error.to_string())
}

#[cfg(test)]
#[path = "sqlite_migrations_test.rs"]
mod tests;
