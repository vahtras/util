test:
	python -m nose -x
debug:
	python -m nose -x --pdb
coverage:
	python -m nose --with-coverage --cover-package util --cover-html tests
