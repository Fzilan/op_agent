from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_OUT_DIR, OPERATOR_FACTS_ROOT


SCHEMA_DIR = OPERATOR_FACTS_ROOT / "schemas"
VALIDATION_DIR = OPERATOR_FACTS_ROOT / "validation"
GOLDEN_DIR = VALIDATION_DIR / "golden"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def check_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def matches_condition(instance: Any, schema: dict) -> bool:
    if "const" in schema:
        return instance == schema["const"]
    expected_type = schema.get("type")
    if expected_type and not check_type(instance, expected_type):
        return False
    if isinstance(instance, dict):
        required = schema.get("required", [])
        for field in required:
            if field not in instance:
                return False
        for key, prop_schema in schema.get("properties", {}).items():
            if key not in instance:
                continue
            if not matches_condition(instance[key], prop_schema):
                return False
    return True


def validate_instance(instance: Any, schema: dict, path: str = "$") -> List[str]:
    errors: List[str] = []

    expected_type = schema.get("type")
    if expected_type and not check_type(instance, expected_type):
        return [f"{path}: expected {expected_type}, got {type(instance).__name__}"]

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']}, got {instance!r}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for field in required:
            if field not in instance:
                errors.append(f"{path}: missing required field {field}")

        properties = schema.get("properties", {})
        for key, value in instance.items():
            if key in properties:
                errors.extend(validate_instance(value, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties", True) is False:
                errors.append(f"{path}: unexpected field {key}")

    if isinstance(instance, list):
        item_schema = schema.get("items")
        if item_schema is not None:
            for idx, item in enumerate(instance):
                errors.extend(validate_instance(item, item_schema, f"{path}[{idx}]"))

    for clause in schema.get("allOf", []):
        cond_schema = clause.get("if")
        then_schema = clause.get("then")
        if cond_schema is None or matches_condition(instance, cond_schema):
            if then_schema is not None:
                errors.extend(validate_instance(instance, then_schema, path))

    return errors


def assert_unique(rows: Iterable[dict], field: str, table_name: str) -> List[str]:
    seen: Dict[str, int] = {}
    errors: List[str] = []
    for idx, row in enumerate(rows):
        value = row.get(field)
        if value in seen:
            errors.append(f"{table_name}: duplicate {field}={value!r} at rows {seen[value]} and {idx}")
        else:
            seen[value] = idx
    return errors


def matches_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(key in actual and matches_subset(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        for expected_item in expected:
            if not any(matches_subset(actual_item, expected_item) for actual_item in actual):
                return False
        return True
    return actual == expected


def find_matching_rows(rows: Iterable[dict], expected: dict) -> List[dict]:
    return [row for row in rows if matches_subset(row, expected)]


def validate_golden_cases(
    golden: dict,
    entry_rows: List[dict],
    unit_rows: List[dict],
    edge_rows: List[dict],
    graph_rows: List[dict],
) -> List[str]:
    errors: List[str] = []
    entry_by_id = {row["entry_id"]: row for row in entry_rows}
    unit_by_id = {row["unit_id"]: row for row in unit_rows}
    edges_by_entry: Dict[str, List[dict]] = {}
    for row in edge_rows:
        edges_by_entry.setdefault(row["entry_id"], []).append(row)
    graph_by_parent: Dict[str, List[dict]] = {}
    for row in graph_rows:
        graph_by_parent.setdefault(row["parent_unit_id"], []).append(row)

    for case in golden.get("cases", []):
        name = case["name"]
        entry_id = case["entry_id"]
        entry_row = entry_by_id.get(entry_id)
        if entry_row is None:
            errors.append(f"golden[{name}]: missing entry_id {entry_id}")
            continue
        if not matches_subset(entry_row, case.get("entry_identity", {})):
            errors.append(f"golden[{name}]: entry_identity mismatch for {entry_id}")

        actual_edges = edges_by_entry.get(entry_id, [])
        if "entry_edge_count" in case and len(actual_edges) != case["entry_edge_count"]:
            errors.append(
                f"golden[{name}]: expected {case['entry_edge_count']} entry edges for {entry_id}, got {len(actual_edges)}"
            )
        for expected_edge in case.get("entry_edges", []):
            if not find_matching_rows(actual_edges, expected_edge):
                errors.append(f"golden[{name}]: missing expected entry edge {expected_edge}")

        for expected_unit in case.get("unit_rows", []):
            if "unit_id" in expected_unit:
                actual_unit = unit_by_id.get(expected_unit["unit_id"])
                if actual_unit is None:
                    errors.append(f"golden[{name}]: missing unit_id {expected_unit['unit_id']}")
                    continue
                if not matches_subset(actual_unit, expected_unit):
                    errors.append(f"golden[{name}]: unit row mismatch for {expected_unit['unit_id']}")
            elif not find_matching_rows(unit_rows, expected_unit):
                errors.append(f"golden[{name}]: missing unit row matching {expected_unit}")

        graph_parent = case.get("graph_parent_unit_id")
        if graph_parent is not None:
            actual_graph = graph_by_parent.get(graph_parent, [])
            if "graph_edge_count" in case and len(actual_graph) != case["graph_edge_count"]:
                errors.append(
                    f"golden[{name}]: expected {case['graph_edge_count']} graph edges for {graph_parent}, got {len(actual_graph)}"
                )
            for expected_graph in case.get("graph_edges", []):
                if not find_matching_rows(actual_graph, expected_graph):
                    errors.append(f"golden[{name}]: missing graph edge {expected_graph}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate refactored MindSpore operator-facts tables.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--schema-dir", type=Path, default=SCHEMA_DIR)
    parser.add_argument("--golden", type=Path, default=GOLDEN_DIR / "ms_facts.golden.json")
    args = parser.parse_args()

    table_specs = [
        ("ms_entry_identity", "entry_id"),
        ("ms_unit_identity", "unit_id"),
        ("ms_entry_unit_edges", "edge_id"),
        ("ms_unit_graph_edges", "graph_edge_id"),
    ]

    schema_errors: List[str] = []
    table_rows: Dict[str, List[dict]] = {}
    for name, unique_field in table_specs:
        schema = load_json(args.schema_dir / f"{name}.schema.json")
        rows = load_jsonl(args.data_dir / f"{name}.jsonl")
        table_rows[name] = rows
        for idx, row in enumerate(rows):
            schema_errors.extend(validate_instance(row, schema, f"{name}[{idx}]"))
        schema_errors.extend(assert_unique(rows, unique_field, name))

    entry_rows = table_rows["ms_entry_identity"]
    unit_rows = table_rows["ms_unit_identity"]
    edge_rows = table_rows["ms_entry_unit_edges"]
    graph_rows = table_rows["ms_unit_graph_edges"]

    entry_ids = {row["entry_id"] for row in entry_rows}
    unit_ids = {row["unit_id"] for row in unit_rows}
    composite_unit_ids = {row["unit_id"] for row in unit_rows if row["unit_type"] == "composite"}

    ref_errors: List[str] = []
    for row in edge_rows:
        if row["entry_id"] not in entry_ids:
            ref_errors.append(f"ms_entry_unit_edges: unknown entry_id {row['entry_id']}")
        if row["unit_id"] not in unit_ids:
            ref_errors.append(f"ms_entry_unit_edges: unknown unit_id {row['unit_id']}")

    for row in graph_rows:
        if row["parent_unit_id"] not in composite_unit_ids:
            ref_errors.append(f"ms_unit_graph_edges: parent_unit_id is not a known composite unit: {row['parent_unit_id']}")
        if row["child_ref_type"] == "unit" and row["child_ref"] not in unit_ids:
            ref_errors.append(f"ms_unit_graph_edges: unknown child unit {row['child_ref']}")
        if row["child_ref_type"] == "public_api" and row["child_ref"] not in entry_ids:
            ref_errors.append(f"ms_unit_graph_edges: unknown public_api child {row['child_ref']}")

    for row in unit_rows:
        if row["unit_type"] != "composite":
            continue
        for leaf in row.get("leaf_units", []):
            leaf_id = leaf["unit_id"]
            if leaf_id not in unit_ids:
                ref_errors.append(f"ms_unit_identity: composite {row['unit_id']} references unknown leaf unit {leaf_id}")

    golden_errors = validate_golden_cases(load_json(args.golden), entry_rows, unit_rows, edge_rows, graph_rows)
    errors = schema_errors + ref_errors + golden_errors

    if errors:
        print(f"validate_ms_facts: FAILED ({len(errors)} issues)")
        for error in errors:
            print(f"- {error}")
        return 1

    print("validate_ms_facts: OK")
    print(f"ms_entry_identity_rows={len(entry_rows)}")
    print(f"ms_unit_identity_rows={len(unit_rows)}")
    print(f"ms_entry_unit_edges_rows={len(edge_rows)}")
    print(f"ms_unit_graph_edges_rows={len(graph_rows)}")
    print(f"golden_cases={len(load_json(args.golden).get('cases', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
