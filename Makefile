.PHONY: setup data pipeline notebooks test dashboard api docker reports
setup:        ## create venv and install everything
	python3.11 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements-dev.txt
data:         ## download Online Retail II from UCI into data/raw
	python scripts/download_data.py
pipeline:     ## clean -> features -> segments -> models -> recommender -> insights
	python scripts/run_pipeline.py
reports:      ## regenerate docs/model_evaluation_report.md from saved metrics
	python scripts/generate_reports.py
notebooks:    ## rebuild and execute the six notebooks
	python scripts/build_notebooks.py
test:
	python -m pytest
dashboard:
	streamlit run app.py
api:
	uvicorn api.main:app --reload
docker:
	docker build -t retailiq-api . && docker run -p 8000:8000 retailiq-api
