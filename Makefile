.PHONY: all test docker test-docker
ALLPY := $(wildcard test/*py) $(wildcard *py)
all: test
test: test/results.txt
.ONESHELL:
test/results.txt: $(ALLPY)
	test -r venv && source venv/bin/activate
	python3 -m pytest --cov=. test/ | tee $@

docker: Dockerfile  $(ALLPY)
	docker build -t eyetracker .
test-docker: docker
	docker run -it eyetracker:latest

