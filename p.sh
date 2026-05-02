#!/bin/bash
git add .
# 매개변수가 있으면 그걸 커밋 메시지로, 없으면 "update"로 저장
msg="${1:-update}"
git commit -m "$msg"
git push origin dev