---
name: skills
description: "Use when writing or updating AIH course PJ experiment docs (PJ1/PJ2/...) with code-report consistency checks, method formulas, ablation notes, and result analysis. PJ1 is a reference case, not a scope limit. Trigger phrases: AIH PJ report, PJ1 report, PJ2 report, method design, bonus, dropout rationale, experiment documentation."
---

# AIH Course PJ Experiment Reporting Skill

## Purpose

Use this skill to produce clear, consistent, and reproducible experiment reports for AIH course projects.

Scope:
- Applies to all PJs in this course (PJ1/PJ2/...).
- PJ1 structure is treated as a reference example.
- For new PJs, reuse the same workflow and adapt section names to the task.

Typical targets:
- Core task documentation (problem, data, method, training, evaluation).
- Bonus design and rationale (regularization, augmentation, architecture changes, etc.).
- Cross-check between code defaults and markdown claims.

## Inputs To Collect First

Before writing, verify these items from the current codebase:
1. Actual network structures used in each part.
2. Training hyperparameters used in scripts (lr, epochs, batch size, split, optimizer).
3. Whether dropout is enabled per task.
4. Which metrics are measured and their latest values.
5. Figure and model artifact filenames.

If the project is not PJ1, also verify:
1. Task-specific metric definitions (for example, MAE/F1/mIoU).
2. Dataset split protocol and seed policy.
3. Any course-required constraints (runtime, model size, prohibited libraries).

## Workflow

1. Read the target markdown report fully.
2. Read implementation files for all tasks and bonus items in the current PJ.
3. Build a consistency table: report claims vs actual code.
4. Update report sections in this order:
- Task definition
- Data and preprocessing
- Method design (structures + formulas)
- Bonus section (if requested)
- Experiment results and analysis
- Deliverables list
5. Re-check for contradictions after editing.

## Recommended Report Skeleton

Use this structure and rename section titles as needed:
1. Task Overview
2. Data and Preprocessing
3. Method Design
4. Training Setup
5. Results
6. Analysis and Error Cases
7. Bonus Work
8. Reproducibility Notes

Keep each section tied to concrete code artifacts.

## Formula Guidelines

When adding formulas:
- Keep notation consistent across sections.
- Prefer compact core equations over derivation-heavy blocks.
- For classification, include softmax + cross-entropy + main gradients.
- For dropout, include forward masking and backprop masking terms.
- For other PJs, include only formulas necessary to explain implementation choices.

Recommended dropout notation:

$$
a_{drop}^{(l)} = \frac{m^{(l)} \odot a^{(l)}}{p},
\quad
m^{(l)} \sim \mathrm{Bernoulli}(p)
$$

$$
\delta^{(l)} = \left(\delta^{(l+1)} W^{(l+1)T} \odot f'(z^{(l)})\right) \odot \frac{m^{(l)}}{p}
$$

## Dropout Decision Rule For Small Models

Use this guidance when writing analysis text:
- If model/task is simple and data regime is not large-noise limited, dropout can reduce effective capacity too much.
- Typical symptom: underfitting and oscillatory training/validation behavior.
- In that case, document the trial and conclude "dropout not enabled" for that task.

Template sentence:
"Because the model and task are relatively simple, enabling dropout reduced effective capacity and led to underfitting with stronger loss oscillation; therefore dropout was not enabled in the final setting for this task."

## Bonus Section Rules (Course-Wide)

For any bonus item, explicitly separate:
1. What was implemented.
2. What was tested but not kept in final configuration.
3. Why the final decision was made.

If no controlled rerun exists, avoid hard performance claims.
Use wording such as:
"Implementation completed; quantitative gain requires controlled rerun under the same seed and split."

## Output Style

- Keep wording concise, technical, and reproducible.
- Distinguish clearly between:
- implemented result (what is in code now)
- comparative claim (requires controlled reruns)
- Avoid inventing metrics that were not re-run.
- If no rerun exists, explicitly state that conclusion is implementation-level, not benchmark-level.

## PJ1 Reference Mapping

Use PJ1 as an example mapping, not a hardcoded template:
- Regression task: state architecture, loss, optimizer, and convergence metric.
- NumPy classifier: state softmax CE setup and class-wise validation behavior.
- CNN classifier: state convolution blocks, classifier head, and validation trajectory.
- Bonus1 dropout: state where enabled and where intentionally disabled.

## Final Consistency Checklist

Before finishing, verify all of the following:
1. Every structure line in markdown matches code.
2. Bonus statements do not conflict with current code state.
3. Result table values are not contradicted by later text.
4. Mentioned files actually exist.
5. Hyperparameters in text match script defaults.
6. "Enabled" and "disabled" decisions in bonus sections are internally consistent.
7. Report statements do not exceed evidence level (implemented vs benchmarked).
