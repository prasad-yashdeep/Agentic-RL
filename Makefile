# Agentic-RL monorepo: dispatch to the subprojects. `make help` lists targets.
include mk/common.mk

.PHONY: help test lint doctor layering

help:         ## this list
> @grep -hE '^[a-z%-]+:.*## ' Makefile | sed -E 's/^([a-z%-]+):[^#]*## /\1\t/'

tb-%:         ## run a terminal-bench target, e.g. make tb-test NODE=node08, make tb-exp
> $(MAKE) -C terminal-bench $* NODE=$(NODE)

swe-%:        ## run a swe-bench target, e.g. make swe-test NODE=node08
> $(MAKE) -C swe-bench $* NODE=$(NODE)

test:         ## both suites, each in its own pytest process on $(NODE) (never on the headnode)
> rc=0; $(MAKE) -C terminal-bench test NODE=$(NODE) || rc=1; $(MAKE) -C swe-bench test NODE=$(NODE) || rc=1; exit $$rc

layering:     ## sarl and tarl never import each other
> $(PY) tools/check_layering.py

lint: layering ## byte-compile everything + whitespace check + import boundaries
> $(PY) -m compileall -q terminal-bench/src terminal-bench/tests terminal-bench/scripts swe-bench/src swe-bench/tests swe-bench/scripts
> git -C $(ARL) diff --check

doctor:       ## stale paths, venv, kernels, Harbor A0 resume consistency (gates make tb-exp)
> bash tools/doctor.sh
