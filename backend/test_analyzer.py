
from code_analyzer import analyze_source_code
from log_analyzer import analyze_error_log

tests = [
    {
        "name": "Python division by zero",
        "type": "source_code",
        "language": "python",
        "text": "result = 10 / 0"
    },
    {
        "name": "Python unsafe eval",
        "type": "source_code",
        "language": "python",
        "text": 'result = eval("2 + 2")'
    },
    {
        "name": "Python safe code",
        "type": "source_code",
        "language": "python",
        "text": "result = 10 + 20"
    },
    {
        "name": "JavaScript unsafe eval",
        "type": "source_code",
        "language": "javascript",
        "text": 'const result = eval("2 + 2");'
    },
    {
        "name": "JavaScript safe code",
        "type": "source_code",
        "language": "javascript",
        "text": "const result = 10 + 20;"
    },
    {
        "name": "Database error",
        "type": "error_log",
        "text": "ERROR: Database connection timeout."
    },
    {
        "name": "Authentication error",
        "type": "error_log",
        "text": "ERROR: Unauthorized access. Authentication failed."
    },
    {
        "name": "Normal application log",
        "type": "error_log",
        "text": "INFO: Application started successfully."
    }
]

print("=" * 60)
print("BUGSENSE AI - ANALYZER FUNCTIONAL TESTS")
print("=" * 60)

passed = 0
failed = 0

for test in tests:
    print(f"\nTEST: {test['name']}")

    try:
        if test["type"] == "source_code":
            result = analyze_source_code(
                test["text"],
                test["language"]
            )
        else:
            result = analyze_error_log(test["text"])

        if not isinstance(result, dict):
            raise TypeError("Analyzer did not return a dictionary")

        print("Execution: PASS")
        print("Severity:", result.get("severity"))
        print("Category:", result.get("category"))
        print("Result:", result)

        passed += 1

    except Exception as error:
        print("Execution: FAIL")
        print("Error:", error)
        failed += 1

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("Executed successfully:", passed)
print("Execution failures:", failed)
print("Total tests:", len(tests))
print("=" * 60)
