#!/bin/bash
git clean -dfx .
python setup.py bdist_wheel --universal
twine upload dist/*
