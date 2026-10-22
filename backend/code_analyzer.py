
import ast
import json
import subprocess
import tempfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

SEVERITY_RANK = {
    "critical": 4,
    "major": 3,
    "normal": 2,
    "minor": 1
}

PRIORITY_MAP = {
    "critical": "Critical",
    "major": "High",
    "normal": "Medium",
    "minor": "Low",
    "unknown": "Unassigned"
}


def make_issue(category, severity, line, message, rule=""):
    return {
        "category": category,
        "severity": severity,
        "line": line,
        "message": message,
        "rule": rule
    }


def python_ast_checks(code):
    issues = []

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [
            make_issue(
                "Syntax Error",
                "major",
                error.lineno,
                error.msg,
                "PY-SYNTAX"
            )
        ]

    for node in ast.walk(tree):

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (
                ast.Div,
                ast.FloorDiv,
                ast.Mod
            )):
                if isinstance(node.right, ast.Constant):
                    if type(node.right.value) in (int, float):
                        if node.right.value == 0:
                            issues.append(make_issue(
                                "Runtime Error",
                                "major",
                                node.lineno,
                                "Division by zero detected.",
                                "PY-DIVZERO"
                            ))

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec"):
                    issues.append(make_issue(
                        "Security Risk",
                        "critical",
                        node.lineno,
                        "Dynamic execution requires security review.",
                        "PY-DYNAMIC"
                    ))

    return issues


def run_ruff(code):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "analysis.py"
        path.write_text(code, encoding="utf-8")

        command = [
            "ruff",
            "check",
            "--isolated",
            "--select",
            "E4,E7,E9,F",
            "--output-format",
            "json",
            str(path)
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                shell=False
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Ruff is not installed. Run: pip install ruff"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ruff analysis timed out.")

        if result.returncode not in (0, 1):
            raise RuntimeError(
                "Ruff failed: " + result.stderr[:300]
            )

        findings = json.loads(result.stdout)
        issues = []

        for item in findings:
            rule = item.get("code", "")
            message = item.get("message", "")
            line = item.get("location", {}).get("row")

            if rule == "F821":
                category = "Undefined Variable"
                severity = "major"
            elif rule.startswith("F"):
                category = "Code Quality"
                severity = "minor"
            else:
                category = "Syntax / Code Quality"
                severity = "normal"

            issues.append(make_issue(
                category,
                severity,
                line,
                message,
                rule
            ))

        return issues


def analyze_python(code):
    ast_issues = python_ast_checks(code)
    ruff_issues = run_ruff(code)

    issues = ast_issues + ruff_issues

    # Avoid duplicate findings from overlapping checks.
    unique = {}
    for issue in issues:
        key = (
            issue["line"],
            issue["category"],
            issue["message"]
        )
        unique[key] = issue

    return list(unique.values())



def analyze_javascript(code):
    # Create temporary files inside the project root
    # so ESLint can analyze them.
    with tempfile.TemporaryDirectory(
        prefix=".bugsense_",
        dir=str(BASE_DIR)
    ) as directory:

        path = Path(directory) / "analysis.jsx"
        path.write_text(code, encoding="utf-8")

        import os

        eslint_path = (
           BASE_DIR / "node_modules" / ".bin" /
           ("eslint.cmd" if os.name == "nt" else "eslint")
        )

        if not eslint_path.exists():
            raise RuntimeError(
                "ESLint not found. Run npm install --save-dev eslint@9"
            )

        command = [
            str(eslint_path),
            "--config",
            str(BASE_DIR / "eslint.config.mjs"),
            "--format",
            "json",
            str(path)
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=20,
                cwd=str(BASE_DIR),
                shell=False
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ESLint analysis timed out.")

        if result.returncode not in (0, 1):
            raise RuntimeError(
                "ESLint failed: " + result.stderr[:300]
            )

        reports = json.loads(result.stdout)

        issues = []

        for report in reports:
            for item in report.get("messages", []):
                rule = item.get("ruleId")
                message = item.get("message", "")
                line = item.get("line")

                # ESLint may report ignored files as warnings
                # without an associated rule ID.
                if rule is None and line is None:
                    raise RuntimeError(
                        "ESLint configuration error: " + message
                    )

                if rule == "no-eval":
                    category = "Security Risk"
                    severity = "critical"

                elif rule == "no-undef":
                    category = "Undefined Variable"
                    severity = "major"

                elif rule is None:
                    category = "Syntax Error"
                    severity = "major"

                else:
                    category = "Code Quality"
                    severity = "minor"

                issues.append(make_issue(
                    category,
                    severity,
                    line,
                    message,
                    rule or "parse-error"
                ))

        return issues



def analyze_source_code(code, language):
    language = language.lower().strip()

    if language == "python":
        issues = analyze_python(code)
        method = "Python AST + Ruff"

    elif language == "javascript":
        issues = analyze_javascript(code)
        method = "ESLint Static Analysis"

    else:
        raise ValueError(
            "Supported languages: python, javascript"
        )

    issues.sort(
        key=lambda issue: (
            issue["line"] if issue["line"] is not None else 0,
            -SEVERITY_RANK.get(issue["severity"], 0)
        )
    )

    if issues:
        highest = max(
            issues,
            key=lambda issue: SEVERITY_RANK.get(
                issue["severity"], 0
            )
        )
        severity = highest["severity"]
        category = highest["category"]
    else:
        severity = "unknown"
        category = "No issues detected"

    return {
        "status": "success",
        "input_type": "source_code",
        "model": "Static Code Analyzer",
        "analysis_method": method,
        "language": language,
        "category": category,
        "severity": severity,
        "priority": PRIORITY_MAP[severity],
        "issues": issues,
        "issue_count": len(issues),
        "explanation": (
            f"Found {len(issues)} potential issues. "
            "Static analysis cannot guarantee bug-free code."
        )
    }
