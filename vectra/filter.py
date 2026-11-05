"""Metadata query filter engine supporting MongoDB-style operators."""

from __future__ import annotations
from typing import Any, Mapping


def matches_filter(metadata: Mapping[str, Any], query_filter: Mapping[str, Any] | None) -> bool:
    """
    Check if a metadata dictionary matches a query filter.

    Supports:
      - Field equality: {"category": "news"}
      - Comparison operators: {"age": {"$gt": 18, "$lte": 65}}
      - Set membership: {"tag": {"$in": ["python", "ai"]}}
      - Substring / containment: {"description": {"$contains": "database"}}
      - Logical operators:
          {"$and": [{"role": "admin"}, {"active": True}]}
          {"$or": [{"category": "a"}, {"category": "b"}]}
          {"$not": {"archived": True}}
    """
    if not query_filter:
        return True

    for key, condition in query_filter.items():
        if key == "$and":
            if not isinstance(condition, (list, tuple)):
                raise ValueError("$and condition must be a list of filters")
            if not all(matches_filter(metadata, sub_filter) for sub_filter in condition):
                return False
        elif key == "$or":
            if not isinstance(condition, (list, tuple)):
                raise ValueError("$or condition must be a list of filters")
            if not any(matches_filter(metadata, sub_filter) for sub_filter in condition):
                return False
        elif key == "$not":
            if not isinstance(condition, dict):
                raise ValueError("$not condition must be a dictionary filter")
            if matches_filter(metadata, condition):
                return False
        else:
            # Field condition
            actual_val = metadata.get(key)
            if not _evaluate_field_condition(actual_val, condition):
                return False

    return True


def _evaluate_field_condition(actual: Any, condition: Any) -> bool:
    """Evaluate a single field condition."""
    if isinstance(condition, dict) and any(k.startswith("$") for k in condition.keys()):
        for op, target in condition.items():
            if op == "$eq":
                if actual != target:
                    return False
            elif op == "$ne":
                if actual == target:
                    return False
            elif op == "$gt":
                if actual is None or actual <= target:
                    return False
            elif op == "$gte":
                if actual is None or actual < target:
                    return False
            elif op == "$lt":
                if actual is None or actual >= target:
                    return False
            elif op == "$lte":
                if actual is None or actual > target:
                    return False
            elif op == "$in":
                if not isinstance(target, (list, tuple, set)):
                    raise ValueError("$in requires a list or set")
                if actual not in target:
                    return False
            elif op == "$nin":
                if not isinstance(target, (list, tuple, set)):
                    raise ValueError("$nin requires a list or set")
                if actual in target:
                    return False
            elif op == "$contains":
                if actual is None:
                    return False
                if isinstance(actual, (str, list, tuple, set, dict)):
                    if target not in actual:
                        return False
                else:
                    return False
            elif op == "$exists":
                if bool(actual is not None) != bool(target):
                    return False
            else:
                raise ValueError(f"Unknown filter operator '{op}'")
        return True
    else:
        # Direct equality check
        return actual == condition
