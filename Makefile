.PHONY: check preflight release-check

check:
	bin/validate.sh --code-only

preflight:
	bin/validate.sh

release-check:
	bin/release-check.sh
