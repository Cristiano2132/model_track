test:
	poetry run pytest

cov:
	poetry run pytest --cov=model_track --cov-report=term-missing