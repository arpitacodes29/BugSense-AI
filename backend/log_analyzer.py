
import re

# Ordered from more specific to more general patterns
ERROR_RULES = [
    {
        "category": "Database Error",
        "patterns": [
            r"sql(?:ite)?exception",
            r"database.*(?:timeout|connection|failed)",
            r"deadlock",
            r"duplicate key",
            r"integrityerror",
            r"psycopg2",
            r"mysql.*error"
        ],
        "severity": "major",
        "priority": "High",
        "explanation": "A database operation or connection appears to have failed."
    },
    {
        "category": "Authentication Error",
        "patterns": [
            r"authentication failed",
            r"unauthorized",
            r"invalid token",
            r"jwt.*expired",
            r"invalid credentials",
            r"permission denied",
            r"access denied"
        ],
        "severity": "major",
        "priority": "High",
        "explanation": "An authentication or authorization problem was detected."
    },
    {
        "category": "Network Error",
        "patterns": [
            r"connection refused",
            r"connection reset",
            r"network.*unreachable",
            r"socket.*timeout",
            r"connecttimeout",
            r"econnrefused",
            r"enotfound",
            r"http.*503"
        ],
        "severity": "major",
        "priority": "High",
        "explanation": "A network connection or service communication problem was detected."
    },
    {
        "category": "Runtime Error",
        "patterns": [
            r"nullpointerexception",
            r"typeerror",
            r"referenceerror",
            r"attributeerror",
            r"indexerror",
            r"keyerror",
            r"zerodivisionerror",
            r"segmentation fault",
            r"outofmemoryerror"
        ],
        "severity": "major",
        "priority": "High",
        "explanation": "The application encountered a runtime exception."
    },
    {
        "category": "File System Error",
        "patterns": [
            r"filenotfounderror",
            r"no such file or directory",
            r"enoent",
            r"disk full",
            r"enospc",
            r"file.*not found",
            r"read-only file system"
        ],
        "severity": "minor",
        "priority": "Medium",
        "explanation": "A file or storage operation appears to have failed."
    }
]

CRITICAL_PATTERNS = [
    r"\bfatal\b",
    r"\bpanic\b",
    r"production.*(?:down|unavailable)",
    r"service.*(?:down|unavailable)",
    r"data loss",
    r"data corruption",
    r"system crash"
]


def analyze_error_log(log_text):
    if not isinstance(log_text, str) or not log_text.strip():
        raise ValueError("Please provide an error log.")

    normalized = log_text.lower()

    matched_rules = []

    for rule in ERROR_RULES:
        matches = [
            pattern
            for pattern in rule["patterns"]
            if re.search(pattern, normalized)
        ]

        if matches:
            matched_rules.append({
                "rule": rule,
                "match_count": len(matches)
            })

    if matched_rules:
        # Prefer the category with the most matching patterns.
        # Original rule order breaks ties.
        best = max(
            matched_rules,
            key=lambda item: item["match_count"]
        )

        selected = best["rule"]

        category = selected["category"]
        severity = selected["severity"]
        priority = selected["priority"]
        explanation = selected["explanation"]

    else:
        category = "Unknown Error"
        severity = "unknown"
        priority = "Unassigned"
        explanation = (
            "No supported error pattern was identified. "
            "Manual investigation is recommended."
        )

    # Explicit critical indicators override default severity.
    critical_detected = any(
        re.search(pattern, normalized)
        for pattern in CRITICAL_PATTERNS
    )

    if critical_detected:
        severity = "critical"
        priority = "Critical"
        explanation += (
            " The log also contains a critical-impact indicator."
        )

    # Extract common exception names
    exceptions = re.findall(
        r"\b[A-Za-z_][A-Za-z0-9_]*"
        r"(?:Exception|Error)\b",
        log_text
    )

    exceptions = list(dict.fromkeys(exceptions))

    return {
        "status": "success",
        "input_type": "error_log",
        "category": category,
        "severity": severity,
        "priority": priority,
        "model": "Rule-Based Log Analyzer",
        "exceptions": exceptions[:10],
        "explanation": explanation,
        "analysis_method": "Pattern matching and heuristic rules"
    }
