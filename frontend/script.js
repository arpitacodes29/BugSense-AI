
"use strict";

// =====================================================
// BUGSENSE AI v2.0 — COMPLETE FRONTEND JAVASCRIPT
// =====================================================

const API_URL = "http://127.0.0.1:5000/api/analyze";
const STORAGE_KEY = "bugsense_history";
const MAX_INPUT_LENGTH = 20000;
const MAX_FILE_SIZE = 100000;
const MAX_HISTORY = 20;

// =====================================================
// DOM ELEMENTS
// =====================================================

const input = document.getElementById("bugInput");
const resultContent = document.getElementById("resultContent");
const historyBody = document.getElementById("historyBody");

const analyzeBtn = document.getElementById("analyzeBtn");
const sampleBtn = document.getElementById("sampleBtn");
const clearBtn = document.getElementById("clearBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

const inputHint = document.getElementById("inputHint");
const tabs = document.querySelectorAll(".tab");

// =====================================================
// APPLICATION STATE
// =====================================================

let activeType = "bug_report";
let selectedLanguage = "javascript";
let isAnalyzing = false;

// =====================================================
// INPUT CONFIGURATION
// =====================================================

const samples = {
  bug_report:
    "Checkout crashes with NullPointerException when " +
    "the cart is empty. Users cannot complete orders.",

  error_log:
    "ERROR: Database connection timeout while saving " +
    "customer records.",

  javascript:
    'var result = 10;\n' +
    'console.log(result);\n' +
    'if (result == "10") {\n' +
    '    eval("console.log(2 + 2)");\n' +
    '}',

  python:
    'def calculate():\n' +
    '    result = 10 / 0\n' +
    '    print(result)\n' +
    '    eval("2 + 2")\n\n' +
    'calculate()'
};

const placeholders = {
  bug_report: "Describe the software bug...",
  source_code: "Paste Python or JavaScript source code...",
  error_log: "Paste application error logs..."
};

const hints = {
  bug_report:
    "Machine learning severity prediction using Advanced SVM.",

  source_code:
    "Static source code analysis for Python and JavaScript.",

  error_log:
    "Rule-based error classification and severity estimation."
};

const inputNames = {
  bug_report: "Bug report",
  source_code: "Source code",
  error_log: "Error log"
};

// =====================================================
// PROGRAMMING LANGUAGE SELECTOR
// =====================================================

const languageContainer = document.createElement("div");
languageContainer.id = "languageContainer";

Object.assign(languageContainer.style, {
  display: "none",
  margin: "12px 0",
  alignItems: "center",
  gap: "12px",
  flexWrap: "wrap"
});

const languageLabel = document.createElement("label");
languageLabel.textContent = "Programming language:";
languageLabel.htmlFor = "languageSelect";

Object.assign(languageLabel.style, {
  color: "#9bbcff",
  fontSize: "14px"
});

const languageSelect = document.createElement("select");
languageSelect.id = "languageSelect";

Object.assign(languageSelect.style, {
  background: "#0d1022",
  color: "#ffffff",
  border: "1px solid #343956",
  borderRadius: "8px",
  padding: "10px 14px",
  cursor: "pointer"
});

[
  ["javascript", "JavaScript"],
  ["python", "Python"]
].forEach(([value, label]) => {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  languageSelect.appendChild(option);
});

languageSelect.value = selectedLanguage;

languageContainer.append(
  languageLabel,
  languageSelect
);

input.parentNode.insertBefore(
  languageContainer,
  input
);

// =====================================================
// SOURCE FILE UPLOAD
// =====================================================

const fileUploadContainer = document.createElement("div");
fileUploadContainer.id = "fileUploadContainer";

Object.assign(fileUploadContainer.style, {
  display: "none",
  margin: "12px 0",
  gap: "10px",
  alignItems: "center",
  flexWrap: "wrap"
});

const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.id = "sourceFileInput";
fileInput.accept = ".py,.js,.jsx";
fileInput.style.display = "none";

const uploadButton = document.createElement("button");
uploadButton.type = "button";
uploadButton.textContent = "Upload source file";
uploadButton.className = "secondary";

Object.assign(uploadButton.style, {
  background: "#252945",
  color: "#a9bbff",
  border: "1px solid #343956",
  borderRadius: "8px",
  padding: "10px 14px",
  cursor: "pointer"
});

const fileNameLabel = document.createElement("span");
fileNameLabel.textContent = "No file selected";

Object.assign(fileNameLabel.style, {
  color: "#9bbcff",
  fontSize: "13px"
});

fileUploadContainer.append(
  fileInput,
  uploadButton,
  fileNameLabel
);

languageContainer.insertAdjacentElement(
  "afterend",
  fileUploadContainer
);

// =====================================================
// HISTORY MANAGEMENT
// =====================================================

function loadHistory() {
  try {
    const saved = JSON.parse(
      localStorage.getItem(STORAGE_KEY) || "[]"
    );

    return Array.isArray(saved)
      ? saved.slice(0, MAX_HISTORY)
      : [];
  } catch (error) {
    console.warn("History loading failed:", error);
    return [];
  }
}

let history = loadHistory();

function saveHistory() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(history)
    );
  } catch (error) {
    console.warn("History saving failed:", error);
  }

  renderHistory();
}

function renderHistory() {
  if (!historyBody) return;

  historyBody.replaceChildren();

  if (history.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");

    cell.colSpan = 6;
    cell.textContent = "No analyses yet.";

    row.appendChild(cell);
    historyBody.appendChild(row);
    return;
  }

  history.forEach((item, index) => {
    const row = document.createElement("tr");

    const values = [
      index + 1,
      item.source || "Unknown",
      item.category || "Not available",
      item.severity || "Unknown",
      item.priority || "Not available",
      item.model || "Unknown"
    ];

    values.forEach(value => {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    });

    historyBody.appendChild(row);
  });
}

// =====================================================
// RESULT DISPLAY HELPERS
// =====================================================

function setResultMessage(message) {
  resultContent.replaceChildren();
  resultContent.className = "result-empty";
  resultContent.textContent = message;
}

function addDetail(container, label, value) {
  const row = document.createElement("div");
  row.className = "result-detail";

  const labelElement = document.createElement("span");
  labelElement.textContent = label;

  const valueElement = document.createElement("strong");
  valueElement.textContent = String(value);

  row.append(
    labelElement,
    valueElement
  );

  container.appendChild(row);
}

function formatValue(value, fallback = "Not available") {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }

  return String(value);
}

function capitalize(value) {
  const text = String(value || "unknown");

  return text.charAt(0).toUpperCase() +
    text.slice(1);
}

// =====================================================
// DISPLAY BACKEND ANALYSIS RESULT
// =====================================================

function displayResult(data, requestedType, requestedLanguage) {
  resultContent.replaceChildren();
  resultContent.className = "result-card";

  const type = requestedType;
  const isBugReport = type === "bug_report";
  const isSourceCode = type === "source_code";
  const isErrorLog = type === "error_log";

  // -------------------------------
  // SEVERITY
  // -------------------------------

  const label = document.createElement("div");
  label.className = "result-label";

  label.textContent = isBugReport
    ? "PREDICTED SEVERITY"
    : "ESTIMATED SEVERITY";

  const severity = document.createElement("div");

  const severityName = String(
    data.severity || "unknown"
  ).toLowerCase();

  severity.className = "result-value";

  const severityClasses = [
    "blocker",
    "critical",
    "major",
    "normal",
    "minor",
    "trivial"
  ];

  if (severityClasses.includes(severityName)) {
    severity.classList.add(
      "severity-" + severityName
    );
  }

  severity.textContent = capitalize(severityName);

  resultContent.append(
    label,
    severity
  );

  // -------------------------------
  // INPUT INFORMATION
  // -------------------------------

  addDetail(
    resultContent,
    "Input",
    inputNames[type] || "Unknown"
  );

  // -------------------------------
  // MODEL / METHOD
  // -------------------------------

  const methodName = isBugReport
    ? "Advanced SVM"
    : isSourceCode
      ? "Static Code Analysis"
      : "Rule-Based Log Analysis";

  addDetail(
    resultContent,
    "Model / Method",
    data.model || data.analysis_method || methodName
  );

  // -------------------------------
  // PROGRAMMING LANGUAGE
  // -------------------------------

  if (isSourceCode) {
    const language = data.language || requestedLanguage;

    addDetail(
      resultContent,
      "Programming language",
      language === "javascript"
        ? "JavaScript"
        : "Python"
    );
  }

  // -------------------------------
  // CATEGORY
  // -------------------------------

  const categoryLabel = isBugReport
    ? "Bug category"
    : isSourceCode
      ? "Issue category"
      : "Error category";

  addDetail(
    resultContent,
    categoryLabel,
    formatValue(data.category)
  );

  // -------------------------------
  // PRIORITY
  // -------------------------------

  addDetail(
    resultContent,
    "Priority",
    formatValue(data.priority)
  );

  // -------------------------------
  // ANALYSIS METHOD
  // -------------------------------

  if (data.analysis_method) {
    addDetail(
      resultContent,
      "Analysis method",
      data.analysis_method
    );
  }

  // -------------------------------
  // EXPLANATION
  // -------------------------------

  if (data.explanation) {
    addDetail(
      resultContent,
      "Explanation",
      data.explanation
    );
  }

  // -------------------------------
  // DETECTED EXCEPTIONS
  // -------------------------------

  if (
    Array.isArray(data.exceptions) &&
    data.exceptions.length > 0
  ) {
    addDetail(
      resultContent,
      "Detected exceptions",
      data.exceptions.join(", ")
    );
  }

  // -------------------------------
  // ERROR LOG FALLBACK
  // -------------------------------

  if (
    isErrorLog &&
    data.category === "Unknown Error" &&
    String(data.severity).toLowerCase() === "unknown"
  ) {
    addDetail(
      resultContent,
      "Detection note",
      "No supported error pattern was identified. " +
      "This does not guarantee that the log is error-free."
    );
  }

  // -------------------------------
  // SOURCE CODE ISSUES
  // -------------------------------

  if (isSourceCode) {
    const issues = Array.isArray(data.issues)
      ? data.issues
      : [];

    addDetail(
      resultContent,
      "Potential issues detected",
      issues.length
    );

    if (issues.length > 0) {
      const heading = document.createElement("div");
      heading.textContent = "DETECTED ISSUES";
      heading.className = "result-label";

      heading.style.marginTop = "22px";
      heading.style.marginBottom = "12px";

      resultContent.appendChild(heading);

      issues.forEach((issue, index) => {
        const issueBox = document.createElement("div");

        Object.assign(issueBox.style, {
          background: "#0d1022",
          border: "1px solid #303653",
          borderRadius: "10px",
          padding: "14px",
          marginBottom: "10px"
        });

        const issueTitle = document.createElement("strong");

        issueTitle.textContent =
          `Issue ${index + 1}: ` +
          (issue.category || "Unknown");

        issueTitle.style.color = "#a9bbff";

        const issueInfo = document.createElement("p");

        issueInfo.textContent =
          `Line ${issue.line ?? "N/A"} | ` +
          `Severity: ${issue.severity || "Unknown"} | ` +
          `Rule: ${issue.rule || "N/A"}`;

        Object.assign(issueInfo.style, {
          fontSize: "13px",
          margin: "8px 0",
          color: "#9bbcff"
        });

        const issueMessage = document.createElement("p");
        issueMessage.textContent = issue.message || "";

        Object.assign(issueMessage.style, {
          margin: "0",
          lineHeight: "1.5"
        });

        issueBox.append(
          issueTitle,
          issueInfo,
          issueMessage
        );

        resultContent.appendChild(issueBox);
      });
    } else {
      addDetail(
        resultContent,
        "Code analysis",
        "No issues detected by the available checks."
      );
    }
  }

  // -------------------------------
  // FINAL STATUS
  // -------------------------------

  addDetail(
    resultContent,
    "Status",
    data.message || "Analysis completed"
  );
}

// =====================================================
// BUTTON STATES
// =====================================================

function updateAnalyzeButton() {
  analyzeBtn.disabled = isAnalyzing;

  analyzeBtn.textContent = isAnalyzing
    ? "Analyzing..."
    : "Analyze";

  sampleBtn.disabled = isAnalyzing;
  clearBtn.disabled = isAnalyzing;

  languageSelect.disabled = isAnalyzing;
  uploadButton.disabled = isAnalyzing;
  fileInput.disabled = isAnalyzing;

  tabs.forEach(tab => {
    tab.disabled = isAnalyzing;
  });
}

// =====================================================
// FILE SELECTION HELPERS
// =====================================================

function resetFileSelection() {
  fileInput.value = "";
  fileNameLabel.textContent = "No file selected";
}

function updateSourceControls() {
  const isSourceCode = activeType === "source_code";

  languageContainer.style.display = isSourceCode
    ? "flex"
    : "none";

  fileUploadContainer.style.display = isSourceCode
    ? "flex"
    : "none";
}

// =====================================================
// TAB SWITCHING
// =====================================================

tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    if (isAnalyzing) return;

    const nextType = tab.dataset.type;

    if (!Object.prototype.hasOwnProperty.call(
      inputNames,
      nextType
    )) {
      return;
    }

    tabs.forEach(item => {
      item.classList.remove("active");
    });

    tab.classList.add("active");

    activeType = nextType;

    input.value = "";
    input.placeholder = placeholders[activeType];
    inputHint.textContent = hints[activeType];

    resetFileSelection();
    updateSourceControls();

    setResultMessage(
      "Paste an input or load a sample, then press Analyze."
    );
  });
});

// =====================================================
// LANGUAGE SWITCHING
// =====================================================

languageSelect.addEventListener("change", () => {
  if (isAnalyzing) return;

  selectedLanguage = languageSelect.value;

  input.value = "";
  resetFileSelection();

  setResultMessage(
    `${capitalize(selectedLanguage)} selected. ` +
    "Paste source code or load a sample."
  );
});

// =====================================================
// LOAD SAMPLE
// =====================================================

sampleBtn.addEventListener("click", () => {
  if (isAnalyzing) return;

  if (activeType === "source_code") {
    input.value = samples[selectedLanguage];
  } else {
    input.value = samples[activeType] || "";
  }

  resetFileSelection();
  input.focus();

  setResultMessage(
    "Sample loaded. Click Analyze to continue."
  );
});

// =====================================================
// CLEAR INPUT
// =====================================================

clearBtn.addEventListener("click", () => {
  if (isAnalyzing) return;

  input.value = "";
  resetFileSelection();

  setResultMessage(
    "Input cleared. Ready for analysis."
  );
});

// =====================================================
// CLEAR HISTORY
// =====================================================

if (clearHistoryBtn) {
  clearHistoryBtn.addEventListener("click", () => {
    if (isAnalyzing) return;

    history = [];
    saveHistory();
  });
}

// =====================================================
// SOURCE FILE UPLOAD EVENTS
// =====================================================

uploadButton.addEventListener("click", () => {
  if (!isAnalyzing && activeType === "source_code") {
    fileInput.click();
  }
});

fileInput.addEventListener("change", async () => {
  if (isAnalyzing || activeType !== "source_code") return;

  const file = fileInput.files[0];

  if (!file) return;

  const extension = file.name
    .split(".")
    .pop()
    .toLowerCase();

  const supportedExtensions = ["py", "js", "jsx"];

  if (!supportedExtensions.includes(extension)) {
    resetFileSelection();
    setResultMessage("Unsupported file type.");
    return;
  }

  if (file.size > MAX_FILE_SIZE) {
    resetFileSelection();

    setResultMessage(
      "File is too large. Maximum upload size is 100 KB."
    );
    return;
  }

  try {
    const content = await file.text();

    if (isAnalyzing || activeType !== "source_code") {
      return;
    }

    if (!content.trim()) {
      throw new Error("The selected file is empty.");
    }

    if (content.length > MAX_INPUT_LENGTH) {
      throw new Error(
        "File exceeds the 20,000 character analysis limit."
      );
    }

    selectedLanguage = extension === "py"
      ? "python"
      : "javascript";

    languageSelect.value = selectedLanguage;

    input.value = content;
    fileNameLabel.textContent = file.name;

    setResultMessage(
      `Loaded ${file.name}. Click Analyze to continue.`
    );
  } catch (error) {
    resetFileSelection();

    setResultMessage(
      "Unable to read file: " + error.message
    );
  }
});

// =====================================================
// MAIN ANALYSIS FUNCTION
// =====================================================

analyzeBtn.addEventListener("click", async () => {
  if (isAnalyzing) return;

  const text = input.value.trim();

  // -------------------------------
  // VALIDATE INPUT
  // -------------------------------

  if (!text) {
    setResultMessage(
      "Please enter some text before analyzing."
    );
    return;
  }

  if (text.length > MAX_INPUT_LENGTH) {
    setResultMessage(
      "Input exceeds the 20,000 character limit."
    );
    return;
  }

  // -------------------------------
  // BUILD REQUEST
  // -------------------------------

  const requestedType = activeType;
  const requestedLanguage = selectedLanguage;

  const payload = {
    input_type: requestedType,
    text: text
  };

  if (requestedType === "source_code") {
    payload.language = requestedLanguage;
  }

  isAnalyzing = true;
  updateAnalyzeButton();

  setResultMessage("Analyzing input...");

  try {
    // -----------------------------
    // SEND REQUEST TO FLASK
    // -----------------------------

    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    let data;

    try {
      data = await response.json();
    } catch {
      throw new Error(
        "The backend returned an invalid JSON response."
      );
    }

    if (!data || typeof data !== "object") {
      throw new Error(
        "The backend returned an unexpected response."
      );
    }

    if (!response.ok || data.status === "error") {
      throw new Error(
        data.error || "Analysis request failed."
      );
    }

    if (
      data.status !== undefined &&
      data.status !== "success"
    ) {
      throw new Error(
        "The backend did not confirm successful analysis."
      );
    }

    // -----------------------------
    // DISPLAY ACTUAL BACKEND RESULT
    // -----------------------------

    displayResult(
      data,
      requestedType,
      requestedLanguage
    );

    // -----------------------------
    // SAVE HISTORY
    // -----------------------------

    history.unshift({
      source: inputNames[requestedType],
      category: data.category || null,
      severity: data.severity || "Unknown",
      priority: data.priority || null,

      model:
        data.model ||
        data.analysis_method ||
        (
          requestedType === "bug_report"
            ? "Advanced SVM"
            : requestedType === "source_code"
              ? "Static Code Analysis"
              : "Rule-Based Log Analysis"
        ),

      language: requestedType === "source_code"
        ? requestedLanguage
        : null,

      issue_count: Array.isArray(data.issues)
        ? data.issues.length
        : null,

      timestamp: new Date().toISOString()
    });

    history = history.slice(0, MAX_HISTORY);

    saveHistory();

  } catch (error) {
    console.error(
      "BugSense analysis error:",
      error
    );

    const message = error instanceof TypeError
      ? "Cannot connect to Flask. Check that the backend is running."
      : error.message;

    setResultMessage(
      "Analysis failed: " + message
    );

  } finally {
    isAnalyzing = false;
    updateAnalyzeButton();
  }
});

// =====================================================
// INITIALIZE APPLICATION
// =====================================================

function initializeApp() {
  input.placeholder = placeholders[activeType];
  inputHint.textContent = hints[activeType];

  updateSourceControls();
  updateAnalyzeButton();
  renderHistory();

  console.log(
    "BugSense AI v2.0 frontend initialized successfully."
  );
}

initializeApp();
