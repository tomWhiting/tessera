# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sentence-transformers==5.7.0",
#   "torch==2.14.0",
#   "transformers==4.57.6",
# ]
# ///

"""Run the shared reference generator with the repository-code compatible stack."""

import sys

from make_reference import main


if __name__ == "__main__":
    sys.exit(main())
