use assert_matches::assert_matches;
use pretty_assertions::assert_eq;
use rusqlite::Connection;

use super::*;

const TEST_MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        sql: "CREATE TABLE migration_test(id INTEGER PRIMARY KEY);",
    },
    Migration {
        version: 2,
        sql: "ALTER TABLE migration_test ADD COLUMN note TEXT;
              CREATE INDEX migration_test_note ON migration_test(note);",
    },
];

fn assert_upgrade_from(starting_version: usize) {
    let mut connection = Connection::open_in_memory().expect("database");
    migrate_to(&mut connection, &TEST_MIGRATIONS[..starting_version]).expect("starting schema");
    if starting_version > 0 {
        connection
            .execute("INSERT INTO migration_test(id) VALUES (1)", [])
            .expect("existing data");
    }

    migrate_to(&mut connection, TEST_MIGRATIONS).expect("upgrade");

    let version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("version");
    let columns = connection
        .prepare("PRAGMA table_info(migration_test)")
        .expect("prepare columns")
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query columns")
        .collect::<Result<Vec<_>, _>>()
        .expect("columns");
    let indexes = connection
        .prepare("PRAGMA index_list(migration_test)")
        .expect("prepare indexes")
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query indexes")
        .collect::<Result<Vec<_>, _>>()
        .expect("indexes");
    let rows: Vec<i64> = connection
        .prepare("SELECT id FROM migration_test")
        .expect("prepare rows")
        .query_map([], |row| row.get(0))
        .expect("query rows")
        .collect::<Result<Vec<_>, _>>()
        .expect("rows");

    assert_eq!(version, 2);
    assert_eq!(columns, ["id", "note"]);
    assert_eq!(indexes, ["migration_test_note"]);
    assert_eq!(
        rows,
        if starting_version > 0 {
            vec![1]
        } else {
            vec![]
        }
    );
}

#[test]
fn upgrades_from_every_schema_version() {
    assert_upgrade_from(0);
    assert_upgrade_from(1);
    assert_upgrade_from(2);
}

#[test]
fn failed_chain_rolls_back_all_versions() {
    const FAILING: &[Migration] = &[
        Migration {
            version: 1,
            sql: "CREATE TABLE migration_test(id INTEGER PRIMARY KEY);",
        },
        Migration {
            version: 2,
            sql: "INSERT INTO missing_table VALUES (1);",
        },
    ];
    let mut connection = Connection::open_in_memory().expect("database");

    assert_matches!(migrate_to(&mut connection, FAILING), Err(_));

    let tables: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master \
             WHERE type = 'table' AND name = 'migration_test'",
            [],
            |row| row.get(0),
        )
        .expect("tables");
    assert_eq!(tables, 0);
    let version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("version");
    assert_eq!(version, 0);
}

#[test]
fn negative_schema_version_is_rejected_without_changing_the_database() {
    let mut connection = Connection::open_in_memory().expect("database");
    connection
        .pragma_update(None, "user_version", -1)
        .expect("invalid version");
    assert_matches!(migrate(&mut connection), Err(StoreError::Persistence(message)) if message.contains("version -1 is negative"));
    let version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("version");
    assert_eq!(version, -1);
}
