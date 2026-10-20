
export default [
  {
    files: ["**/*.js", "**/*.jsx"],

    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      parserOptions: {
        ecmaFeatures: {
          jsx: true
        }
      },
      globals: {
        console: "readonly",
        window: "readonly",
        document: "readonly",
        fetch: "readonly",
        module: "readonly",
        require: "readonly",
        process: "readonly"
      }
    },

    rules: {
      "no-undef": "error",
      "no-eval": "error",
      "eqeqeq": ["warn", "always"],
      "no-var": "warn",
      "no-unused-vars": "warn",
      "no-debugger": "warn",
      "no-constant-condition": "warn"
    }
  }
];
