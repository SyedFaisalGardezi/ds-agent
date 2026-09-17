from __future__ import annotations

import sys

from agent.connectors.flatfile_conn import FlatFileConnector
from agent.connectors.postgres_conn import PostgresConnector
from agent.connectors.snowflake_conn import SnowflakeConnector


def validate_connector_interface(cls: type) -> list[str]:
    errors = []
    required = ["connect", "list_schemas", "describe_table", "query", "estimate_cost", "detect_pii"]
    for method in required:
        if not hasattr(cls, method):
            errors.append(f"{cls.__name__} missing method: {method}")
    return errors


def main() -> None:
    connectors = [SnowflakeConnector, PostgresConnector, FlatFileConnector]
    all_errors = []
    for cls in connectors:
        errors = validate_connector_interface(cls)
        all_errors.extend(errors)

    if all_errors:
        print("Schema validation FAILED:")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    print("Schema validation PASSED.")


if __name__ == "__main__":
    main()
