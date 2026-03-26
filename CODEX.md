# AGENTS.md

## Project Identity
This repository is a capstone project for a healthcare application titled:

**인바디 기반 자취생 맞춤형 일일 권장섭취량 추천 헬스케어 앱**

The repository contains two connected projects:
1. **ML / recommendation / health logic project**
2. **App project using the trained or implemented recommendation/prediction logic**

This project is not a medical diagnosis system. It is a capstone prototype for research and educational purposes.

---

## Core Project Goal
The purpose of this project is to use:
- InBody body composition information
- user profile / activity / goal information
- food nutrition information

to calculate:
- daily recommended intake
- remaining intake after meals
- convenience-store-friendly meal combinations for students living alone

The system should focus on **realistic eating environments** such as convenience stores, lunch boxes, instant meals, and irregular meal patterns.

---

## What Must Be Preserved
When modifying this repository, always preserve the following core intent:

1. **Personalized nutrition guidance based on body composition**
2. **Student-friendly convenience store meal recommendation**
3. **Practical usability over theoretical perfection**
4. **Simple and understandable UX**
5. **Consistent outputs for the same inputs**
6. **Separation between research logic and app/UI logic**

Do not turn this project into a generic calorie app.
Do not shift the core target away from **students living alone / convenience-store meal scenarios**.

---

## Repository Assumptions
Assume the repository is organized around:
- `ml/` → nutrition logic, recommendation engine, modeling, preprocessing, evaluation
- `app/` → mobile/web app UI and interaction
- `data/` → local-only raw or processed data paths
- `docs/` → architecture, report, references, diagrams, documentation

If structure changes are needed, prefer minimal, explainable changes.

---

## Scope Rules
Prioritize implementation in this order:

### Highest priority
- user input flow for body / InBody / goal information
- daily intake calculation logic
- convenience-store food nutrition lookup
- remaining nutrient calculation
- recommendation result generation
- result explanation / feedback UI

### Medium priority
- scenario-based recommendation improvements
- better filtering / ranking logic
- reusable data models
- validation and test coverage

### Lower priority
- overly complex infrastructure
- unnecessary refactors
- premature optimization
- features unrelated to the capstone evaluation scope

---

## Domain Constraints
This repository handles health-related information. Therefore:

1. Never present outputs as medical diagnosis.
2. Never claim treatment, prescription, or disease certainty.
3. Always preserve disclaimer-style wording where relevant.
4. Prefer wording like:
   - "추천"
   - "가이드"
   - "참고용"
   - "건강관리 보조"
5. Avoid wording like:
   - "진단"
   - "확정"
   - "치료"
   - "의학적 판정"

---

## Data Rules
Data safety is mandatory.

### Never commit:
- raw KNHANES CSV files
- processed person-level health datasets
- exported sensitive health records
- secrets, API keys, `.env` files
- model artifacts unless explicitly intended and approved

### Always prefer:
- scripts that reproduce preprocessing
- dummy/sample data
- documentation for expected data paths
- ignored local storage for raw datasets

If a task would require adding real raw healthcare data to git, do **not** do that.

---

## Copyright / Source Rules
This project must respect:
- public data usage conditions
- citation / attribution obligations
- copyright of nutrition datasets, papers, and external materials

When generating docs or code comments:
- do not copy long copyrighted text
- summarize instead
- keep source references in docs when needed

---

## Engineering Style
When working on code in this repository:

1. Keep changes small and reviewable.
2. Do not rewrite unrelated modules.
3. Maintain clear separation of concerns.
4. Prefer readable and explainable logic over clever shortcuts.
5. Add or update README/docs when behavior changes.
6. Preserve folder purpose boundaries (`ml/`, `app/`, `docs/`, `scripts/`).
7. Add comments only where they improve understanding.
8. Write predictable code that teammates can continue easily.

---

## Recommendation Logic Guidelines
For recommendation-related code:

- prioritize consistency and interpretability
- calculations should be traceable from input to output
- if multiple recommendations are possible, prefer practical combinations
- avoid extreme diet logic
- consider already consumed nutrients before suggesting the next meal
- prefer recommendations that users can realistically buy and eat

When uncertain, choose realism and user comprehension over algorithmic complexity.

---

## UX Guidelines
For app/UI work:

- optimize for quick understanding
- show remaining calories/macros clearly
- make meal recommendation results easy to compare
- avoid overly technical health terminology unless explained
- keep input flow short and intuitive
- design around actual user scenarios, not abstract dashboards

---

## Testing Expectations
When adding or changing features, prefer tests or checks for:

- same input → same result consistency
- intake calculation correctness
- meal recommendation logic validity
- edge cases for missing/partial user input
- stable UI behavior for main user flows

If no formal test framework exists, at least add lightweight validation logic or documented manual test steps.

---

## Team Context
Use the following collaboration assumptions:

- This is a team capstone repository.
- Changes should remain understandable to all teammates.
- Documentation quality matters because final reporting and presentation are part of the project.
- Prefer outputs that help both implementation and report writing.

---

## Documentation Duties
Whenever a meaningful feature is added or changed, also check whether one of these should be updated:
- root `README.md`
- `ml/README.md`
- `app/README.md`
- `docs/architecture.md`
- API/data-flow explanation docs
- test notes / scenario docs

---

## Branch / Commit Guidance
Prefer branches like:
- `feature/intake-calculation`
- `feature/recommendation-engine`
- `feature/app-input-flow`
- `fix/nutrition-calculation`
- `docs/report-alignment`

Prefer commit prefixes like:
- `feat:`
- `fix:`
- `docs:`
- `refactor:`
- `test:`
- `chore:`

---

## Things To Avoid
Do not:
- replace the project’s core theme with a generic health tracker
- add real raw health datasets to version control
- introduce unrequested heavy infrastructure
- make medical claims
- break the separation between recommendation logic and app presentation
- remove disclaimers or documentation about safe usage
- make large speculative changes without explaining them

---

## Default Operating Behavior For Agents
When starting work in this repository, first:

1. read `README.md`
2. inspect `ml/` and `app/` separately
3. identify whether the task belongs to:
   - nutrition logic
   - data preprocessing
   - recommendation engine
   - UI/app flow
   - documentation
4. make the smallest correct change
5. update docs if behavior changed

If requirements are ambiguous, prefer alignment with:
- body-composition-based personalized recommendation
- convenience-store meal recommendation for students living alone
- practical capstone deliverables
- safety and data handling constraints

---

## Short Project Summary For Agents
This repository implements a capstone healthcare prototype that calculates personalized daily intake and recommends convenience-store-friendly meals using InBody/body data, user goals, and nutrition information. The project must remain practical, explainable, privacy-conscious, and clearly non-diagnostic.
