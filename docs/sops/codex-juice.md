# Codex CLI Metadata SOPs

- **Title**: SOP: Check assistant "juice" budget for this run
  **Prereqs**: Codex CLI harness that exposes `# Juice: <N>` in the system prompt metadata (this environment)
  **Steps**:
  - Ask the assistant: `your juice is?`
  - Read the assistant's reply (example: `768`)
  **Expected Result**: A numeric "juice" value for the current run.
  **Troubleshooting**: If the assistant cannot report a value, treat "juice" as unspecified for that environment.
  **References**: N/A

