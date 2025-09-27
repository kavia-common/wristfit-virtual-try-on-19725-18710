#!/bin/bash
cd /home/kavia/workspace/code-generation/wristfit-virtual-try-on-19725-18710/api_hand_detection
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

