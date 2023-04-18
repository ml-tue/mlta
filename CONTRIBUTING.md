# Contributing

To contribute to the MLTA, please follow these guidelines to make code reviews and merging hassle-free. Mainly, before creating a pull request, make your contribution fulfill the following requirements:

1. The code formatting follows the [black](https://github.com/psf/black) style guidelines. This is easily checked by setting up a pre-commit hook, see below.
2. The import order follows the one prescribed by [isort](https://github.com/pycqa/isort). This is easily checked by setting up a pre-commit hook, see below.
3. If applicable: All tests are passing. 
4. If applicable: Your contribution does not lower the test coverage.

## Setup Pre-Commit Hooks
This project uses [`pre-commit`](https://pre-commit.com) for managing Git's wonderful pre-commit hooks. To set it up, follow the following steps (all in the repository root folder):

1. Install `pre-commit`: run `pip install pre-commit`
2. Set up the hooks: run `pre-commit`
3. That's it!