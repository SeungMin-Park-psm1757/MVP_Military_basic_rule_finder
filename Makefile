.PHONY: sample ingest app test smoke

sample:
	python scripts/build_sample_corpus.py

ingest:
	python scripts/ingest_to_chroma.py --input data/sample/processed/sample_documents.jsonl

app:
	streamlit run streamlit_app.py

test:
	pytest -q

smoke:
	python scripts/run_smoke_checks.py
