test:
	#python -m pytest --cov util --cov-report=html tests
	python -m nose --with-coverage --cover-package util tests
