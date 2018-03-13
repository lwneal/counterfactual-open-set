#!/bin/bash

echo "Baseline and OpenMax results (non-generative):"
cat `find evaluations/ | grep json | sort | head -1`

echo "Results with generated data:"
cat `find evaluations/ | grep json | sort | tail -1`
