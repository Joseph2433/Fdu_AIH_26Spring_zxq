---
agent: agent
description: "Generate or update AIH course PJ experiment report sections with code-consistency checks and evidence-aware conclusions. Use for PJ1/PJ2/... report writing."
---

# AIH PJ Report Update Prompt

Use the skill `pj1-experiment-reporting` as the primary workflow.

## Inputs

- PJ name or id: `${input:pj_name:例如 PJ1}`
- Target report file: `${input:report_file:例如 pj1/实验文档.md}`
- Scope to update: `${input:scope:例如 方法设计+Bonus+结果分析}`
- Code folders/files to verify: `${input:code_scope:例如 pj1/p1, pj1/p2}`
- Bonus items involved: `${input:bonus_scope:例如 bonus1(dropout)}`
- Evidence level: `${input:evidence_level:implemented-only 或 benchmarked}`

## Task

Based on the current codebase and target markdown report:

1. Read the report and all relevant implementation files first.
2. Build a concise consistency checklist between code defaults and report claims.
3. Update only the requested scope.
4. Keep formulas and symbols consistent with existing sections.
5. For bonus decisions, clearly separate:
- what was implemented
- what was tested but not kept
- why the final decision was made
6. If evidence level is `implemented-only`, do not claim quantitative improvement without controlled reruns.
7. Preserve existing style and section numbering whenever possible.

## Output Format

Return:
1. A short change summary.
2. Exact files edited.
3. Any unresolved assumptions.
4. Optional next-step rerun suggestions only if needed.

## Guardrails

- Do not fabricate metrics.
- Do not state "improved" unless there is explicit comparable evidence.
- Keep language technical, concise, and reproducible.
- Ensure final text does not contradict code state.
