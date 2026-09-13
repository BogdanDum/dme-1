.DEFAULT_GOAL := help

PYTHON ?= python

.PHONY: help check inventory run train test syndromes google qasm

help:
	@echo "Student pipeline commands"
	@echo ""
	@echo "  make check      Verify object-store and PostgreSQL connections"
	@echo "  make inventory  List the supplied course data files"
	@echo "  make run        Run Part I: verify Bronze, build Silver, write results"
	@echo "  make train      Train/evaluate models from the required ML input tables"
	@echo "  make test       Run automated tests"
	@echo ""
	@echo "Single-source runs, for iterating on one parser:"
	@echo "  make syndromes  Run Part I for qec_syndromes only"
	@echo "  make google     Run Part I for google_qec only"
	@echo "  make qasm       Run Part I for qasmbench only"

check:
	$(PYTHON) -m quantum_lake_student.cli check

inventory:
	$(PYTHON) -m quantum_lake_student.cli inventory

run:
	$(PYTHON) -m quantum_lake_student.cli run

train:
	$(PYTHON) -m quantum_lake_student.cli train

test:
	$(PYTHON) -m pytest

syndromes:
	$(PYTHON) -m quantum_lake_student.cli run --source qec_syndromes

google:
	$(PYTHON) -m quantum_lake_student.cli run --source google_qec

qasm:
	$(PYTHON) -m quantum_lake_student.cli run --source qasmbench
