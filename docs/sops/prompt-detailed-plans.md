# Prompting SOPs

- **Title**: SOP: Ask for highly detailed step-by-step plans
  **Prereqs**: N/A
  **Steps**:
  - State that the assistant must use the plan tool (for example, `update_plan`) and require a minimum number of steps.
  - Require each step to include fields like goal, inputs, outputs, dependencies, and validation, and prohibit skipping.
  - Require plan updates after each completed step and forbid moving ahead without a status change.
  - Ask for explicit assumptions and to note any higher-priority policy conflicts before proceeding.
  - Use a concrete prompt template and tune the step count and fields as needed.
  **Expected Result**: The assistant returns a multi-step plan with explicit gating and then executes it step by step.
  **Troubleshooting**: If the assistant refuses or shortens the plan, increase the minimum step count or relax constraints that conflict with system or developer policies.
  **References**: N/A

