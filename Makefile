build:
	python3 -m build

upload:
	twine upload dist/*
