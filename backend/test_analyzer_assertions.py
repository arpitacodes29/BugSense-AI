
from code_analyzer import analyze_source_code
from log_analyzer import analyze_error_log

tests = [
    {
        "name": "Python division by zero",
        "type": "source_code",
        "language": "python",
        "text": "result = 10 / 0",
        "expected_rule": "PY-DIVZERO",
    },
    {
        "name": "Python unsafe eval",
        "type": "source_code",
        "language": "python",
        "text": 'result = eval("2 + 2")',
        "expected_rule": "PY-DYNAMIC",
    },
    {
        "name": "Python safe code",
        "type": "source_code",
        "language": "python",
        "text": "result = 10 + 20",
        "expected_rules": [],
    },
    {
        "name": "JavaScript unsafe eval",
        "type": "source_code",
        "language": "javascript",
        "text": 'const result = eval("2 + 2");',
        "expected_rule": "no-eval",
    },
    {
        "name": "JavaScript safe code",
        "type": "source_code",
        "language": "javascript",
        "text": "console.log(10 + 20);",
        "expected_rules": [],
    },
    {
        "name": "Database error",
        "type": "error_log",
        "text": "ERROR: Database connection timeout.",
        "expected_category": "Database Error",
    },
    {
        "name": "Authentication error",
        "type": "error_log",
        "text": "ERROR: Unauthorized access. Authentication failed.",
        "expected_category": "Authentication Error",
    },
    {
        "name": "Normal application log",
        "type": "error_log",
        "text": "INFO: Application started successfully.",
        "expected_category": None,
    },
]

passed = 0
failed = 0

for test in tests:
    name = test["name"]

    try:
        if test["type"] == "source_code":
            result = analyze_source_code(
                test["text"],
                test["language"]
            )

            issues = result.get("issues", [])
            actual_rules = [
                issue.get("rule") for issue in issues
            ]

            if "expected_rule" in test:
                correct = test["expected_rule"] in actual_rules
            else:
                correct = actual_rules == test["expected_rules"]

            actual = actual_rules

        else:
            result = analyze_error_log(test["text"])
            actual = result.get("category")

            if test["expected_category"] is None:
                correct = (
                    actual in (
                        None,
                        "Unknown",
                        "Unknown Error",
                        "No Error",
                        "No issues detected",
                        "Unclassified"
                    )
                    and result.get("severity") == "unknown"
                    and result.get("exceptions") == []
                )

        if correct:
            passed += 1
            print(f"PASS | {name}")
        else:
            failed += 1
            print(f"FAIL | {name}")
            print("Actual:", actual)
            print("Full result:", result)

    except Exception as error:
        failed += 1
        print(f"ERROR | {name}: {error}")

print("\n" + "=" * 55)
print("BUGSENSE AI - ANALYZER CORRECTNESS SUMMARY")
print("=" * 55)
print("Passed:", passed)
print("Failed:", failed)
print("Total:", len(tests))
