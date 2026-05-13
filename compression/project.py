#!/usr/bin/env python3
#
# Backward-compatible entrypoint for fixed CS projection generation.
#

from compression.fixed_cs import DEFAULT_CONFIG, METHODS as PROJECTED_METHODS, main, run


if __name__ == "__main__":
    raise SystemExit(main())
