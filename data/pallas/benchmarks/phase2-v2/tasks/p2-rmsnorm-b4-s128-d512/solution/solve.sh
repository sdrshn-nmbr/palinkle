#!/bin/bash
set -euo pipefail
cp /solution/kernel.py /app/kernel.py
git -C /app add kernel.py
git -C /app -c user.name=oracle -c user.email=oracle@local commit -m solution
