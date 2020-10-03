.PHONY: all clean tests

SOURCES := $(shell git ls-files '*.py')

all: clean tests
clean:
	rm -f *.bin
tests:
	pylint $(SOURCES)
	mypy $(SOURCES)
	black --check $(SOURCES)
	nosetests -v -s