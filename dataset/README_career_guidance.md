# Career guidance dataset notes

## Files
- `career_data.xlsx` — training data for the ML career model (P1–P8 + 8 aptitude columns → Job profession).
- `career_questions.json` — **8 career-related aptitude questions** seeded into MongoDB `career_questions`.
- `pakistan_career_scope.json` — Pakistan job-portal scope % (LinkedIn / Rozee / Mustakbil / other) seeded into MongoDB `career_scope`.

## Skill columns (do not rename without retrain)
The model still uses these column names from `career_data.xlsx`:
Linguistic, Musical, Bodily, Logical - Mathematical, Spatial-Visualization, Interpersonal, Intrapersonal, Naturalist.

UI questions are rewritten to be **career aptitude** questions, but answers still map into those same columns so predictions stay valid without a full retrain.

## APIs
- `GET /career-questions` — questions from DB (fallback: JSON file)
- `GET|POST /career-scope` — Pakistan scope % for a career title
- `POST /predict-career` — marks + aptitude → top careers
