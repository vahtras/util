test:
	python -m pytest -x --tb=line
debug:
	python -m pytest -x --pdb
coverage:
	python -m pytest --cov=util --cov-report=html tests
dists:
	python -m pep517.build .
