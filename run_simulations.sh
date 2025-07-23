#!/bin/bash
for i in {1..20}; do
    echo "========== Simulation iteration $i =========="
    python3 main/env1.py
done

