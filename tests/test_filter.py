import pytest
from vectra.filter import matches_filter


def test_matches_filter_empty():
    assert matches_filter({"category": "ai"}, None) is True
    assert matches_filter({"category": "ai"}, {}) is True


def test_matches_filter_exact_match():
    meta = {"category": "ai", "active": True, "stars": 10}
    assert matches_filter(meta, {"category": "ai"}) is True
    assert matches_filter(meta, {"category": "web"}) is False


def test_matches_filter_comparison():
    meta = {"stars": 42, "ratio": 0.85}
    assert matches_filter(meta, {"stars": {"$gt": 40, "$lte": 50}}) is True
    assert matches_filter(meta, {"stars": {"$lt": 40}}) is False
    assert matches_filter(meta, {"stars": {"$gte": 42}}) is True
    assert matches_filter(meta, {"stars": {"$ne": 40}}) is True
    assert matches_filter(meta, {"stars": {"$eq": 42}}) is True


def test_matches_filter_membership():
    meta = {"tags": ["python", "ai", "vector"], "role": "admin"}
    assert matches_filter(meta, {"role": {"$in": ["admin", "superadmin"]}}) is True
    assert matches_filter(meta, {"role": {"$nin": ["guest", "user"]}}) is True
    assert matches_filter(meta, {"tags": {"$contains": "python"}}) is True
    assert matches_filter(meta, {"tags": {"$contains": "rust"}}) is False


def test_matches_filter_logical():
    meta = {"category": "ai", "stars": 50, "verified": True}
    flt_and = {"$and": [{"category": "ai"}, {"stars": {"$gte": 50}}]}
    assert matches_filter(meta, flt_and) is True

    flt_or = {"$or": [{"category": "web"}, {"verified": True}]}
    assert matches_filter(meta, flt_or) is True

    flt_not = {"$not": {"category": "web"}}
    assert matches_filter(meta, flt_not) is True

    flt_not_fail = {"$not": {"category": "ai"}}
    assert matches_filter(meta, flt_not_fail) is False
