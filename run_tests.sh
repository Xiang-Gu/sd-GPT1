#!/usr/bin/env bash
set -e

venv/bin/python -m unittest discover -s tests -v
