
import json
from app import app

client = app.test_client()

tests = [
    {
        "name": "Health check",
        "method": "GET",
        "url": "/api/health",
        "expected_status": 200,
        "expected_fields": ["status", "model_loaded"]
    },
    {
        "name": "Bug severity prediction",
        "method": "POST",
        "url": "/api/analyze",
        "data": {
            "input_type": "bug_report",
            "text": "Application crashes when uploading a file."
        },
        "expected_status": 200,
        "expected_fields": ["severity", "model"]
    },
    {
        "name": "Python static analysis",
        "method": "POST",
        "url": "/api/analyze",
        "data": {
            "input_type": "source_code",
            "language": "python",
            "text": "result = 10 / 0"
        },
        "expected_status": 200,
        "expected_fields": ["issues", "issue_count"]
    },
    {
        "name": "JavaScript static analysis",
        "method": "POST",
        "url": "/api/analyze",
        "data": {
            "input_type": "source_code",
            "language": "javascript",
            "text": 'eval("2 + 2");'
        },
        "expected_status": 200,
        "expected_fields": ["issues", "issue_count"]
    },
    {
        "name": "Error log analysis",
        "method": "POST",
        "url": "/api/analyze",
        "data": {
            "input_type": "error_log",
            "text": "ERROR: Database connection timeout."
        },
        "expected_status": 200,
        "expected_fields": ["category", "severity"]
    },
    {
        "name": "Empty input",
        "method": "POST",
        "url": "/api/analyze",
        "data": {
            "input_type": "bug_report",
            "text": ""
        },
        "expected_status": 400,
        "expected_fields": ["error"]
    },
    {
        "name": "Unsupported language",
        "method": "POST",
        "url": "/api/analyze",
        "data": {
            "input_type": "source_code",
            "language": "java",
            "text": "System.out.println(1);"
        },
        "expected_status": 400,
        "expected_fields": ["error"]
    },
    {
        "name": "Unsupported input type",
        "method": "POST",
        "url": "/api/analyze",
        "data": {
            "input_type": "unknown",
            "text": "Test"
        },
        "expected_status": 400,
        "expected_fields": ["error"]
    },
    {
        "name": "Oversized input",
        "method": "POST",
        "url": "/api/analyze",
        "data": {
            "input_type": "bug_report",
            "text": "A" * 20001
        },
        "expected_status": 400,
        "expected_fields": ["error"]
    },
    {
        "name": "Missing endpoint",
        "method": "GET",
        "url": "/api/does-not-exist",
        "expected_status": 404,
        "expected_fields": ["error"]
    }
]

passed = 0
failed = 0

print("=" * 60)
print("BUGSENSE AI - API INTEGRATION TESTS")
print("=" * 60)

for test in tests:
    try:
        if test["method"] == "GET":
            response = client.get(test["url"])
        else:
            response = client.post(
                test["url"],
                json=test["data"]
            )

        result = response.get_json()

        status_ok = (
            response.status_code == test["expected_status"]
        )

        fields_ok = (
            isinstance(result, dict)
            and all(
                field in result
                for field in test["expected_fields"]
            )
        )

        correct = status_ok and fields_ok

        if correct:
            passed += 1
            print("PASS |", test["name"])
        else:
            failed += 1
            print("FAIL |", test["name"])
            print("HTTP status:", response.status_code)
            print("Response:", json.dumps(result, indent=2))

    except Exception as error:
        failed += 1
        print("ERROR |", test["name"], "|", error)

print("\n" + "=" * 60)
print("API TEST SUMMARY")
print("=" * 60)
print("Passed:", passed)
print("Failed:", failed)
print("Total:", len(tests))
