# Everything here runs locally and is free of charge.
.PHONY: help install data synth retrieval-eval judge-eval ablation eval api lint typecheck test docker

help:
	@echo "install        Install the package + dev extras (editable)"
	@echo "data           Rebuild the conversation eval set from Bitext"
	@echo "synth          Rebuild the synthetic rare-violation set"
	@echo "retrieval-eval Hit@k / recall@k / MRR, BM25 vs dense"
	@echo "judge-eval     Judge agreement with human gold (Cohen's kappa)"
	@echo "ablation       Retrieved top-k context vs full-KB stuffing -> decision"
	@echo "eval           retrieval-eval + judge-eval + ablation"
	@echo "api            Serve the FastAPI service on :8000"
	@echo "lint | typecheck | test   ruff | mypy | pytest"
	@echo "docker         Build & run the API via docker compose"

# Backends for the eval targets. Override on a GPU box, e.g.:
#   make judge-eval JUDGE_BACKENDS="heuristic transformers" RETRIEVER=dense
RETRIEVER ?= dense
JUDGE_BACKENDS ?= heuristic transformers

install:
	pip install -e ".[dev,dense,llm,data,trace]"

data:
	python -m qaudit.data.load --build

synth:
	python -m qaudit.synth.generate --build

retrieval-eval:
	python -m qaudit.eval.retrieval_eval --retrievers bm25 dense --k 1 3 5

judge-eval:
	QAUDIT_RETRIEVER=$(RETRIEVER) python -m qaudit.eval.judge_eval --backends $(JUDGE_BACKENDS)

ablation:
	QAUDIT_RETRIEVER=$(RETRIEVER) python -m qaudit.eval.ablation --backend transformers --modes retrieved full

eval: retrieval-eval judge-eval ablation

api:
	uvicorn qaudit.api.main:app --reload --port 8000 --app-dir src

lint:
	ruff check src tests

typecheck:
	mypy src

test:
	pytest -q

docker:
	docker compose up --build
