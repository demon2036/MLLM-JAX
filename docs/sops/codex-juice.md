# Codex “Juice” SOPs

- **Title**: SOP: Read the current run’s “juice” (token budget)
  **Prereqs**: Running inside a Codex CLI/chat harness that prints a `Juice: <number>` line in the system/session metadata
  **Steps**:
  - In the current chat/session metadata (system prompt), locate the line that looks like `Juice: 32000`
  - Read the integer after `Juice:`; treat it as the maximum token budget for the run
  **Expected Result**: You can state the run’s juice value (e.g., `32000`)
  **Troubleshooting**: If no `Juice:` line is present, your runner may not expose it; check your harness UI/logs for an equivalent “token budget” field
  **References**: N/A
