# Visit with Us — Wellness Tourism MLOps

A pre-contact purchase propensity pipeline using the supplied tourism.csv.

## Repository structure
- tourism_project/data/tourism.csv: supplied dataset.
- tourism_project/model_building/: registration, preparation and model search.
- tourism_project/deployment/: Streamlit app, model, metadata and dependencies.
- .github/workflows/pipeline.yml: four-job artifact-based workflow.
- tests/: schema, leakage, persistence, model quality and Streamlit smoke checks.
- reports/: recorded experiments, metrics and evaluation figures.

## Reproduce (Python 3.12)
```bash
python -m pip install -r requirements-ci.txt
python -m tourism_project.model_building.data_register
python -m tourism_project.model_building.prep
python -m tourism_project.model_building.train --require-mlflow
python -m unittest discover -s tests -v
python -m streamlit run tourism_project/deployment/app.py
```

## Deployment
The workflow passes splits between jobs using Actions artifacts, logs all model
candidates to MLflow, tests the model and app, and commits a validated model to main.
A successful cloud run must be observed in Actions; the presence of source alone
is not proof of execution. Main-branch publishing must be permitted in this course repo.

Create the Streamlit Community Cloud app with branch main and entrypoint
tourism_project/deployment/app.py. Select Python 3.12. No app secrets are needed.
Optional Hugging Face deployment uses a public Docker Space and port 7860.

## Model limitations
Only pre-contact attributes are used. Pitch outcomes/follow-ups and identifiers
are excluded. Matching profiles are grouped across holdout and CV partitions.
Preprocessing is fitted inside CV; the F2 threshold uses training OOF predictions.
Historical package purchases do not establish wellness-specific demand or causal
marketing lift. Validate a new campaign, calibration, costs and subgroup outcomes.
Preserve fresh temporal test data as new labels arrive. No operational drift service
is implemented by this training workflow alone.

## Documentation
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.github.com/en/actions/tutorials/store-and-share-data
- https://mlflow.org/docs/latest/tracking/
- https://huggingface.co/docs/hub/spaces-sdks-docker
