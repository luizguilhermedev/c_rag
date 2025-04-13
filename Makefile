sync:
	uv pip sync requirements/prd.txt

lint:
	ruff check --fix
	ruff format

reqs:
	uv pip compile --generate-hashes pyproject.toml -o requirements/prd.txt
