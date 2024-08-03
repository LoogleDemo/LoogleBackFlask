#!/usr/bin/env bash

PROJECT_ROOT="/home/ubuntu/flaskapp"
APP_LOG="$PROJECT_ROOT/application.log"
ERROR_LOG="$PROJECT_ROOT/error.log"
DEPLOY_LOG="$PROJECT_ROOT/deploy.log"

TIME_NOW=$(date +%c)

# 애플리케이션 시작
echo "$TIME_NOW > Flask 애플리케이션 시작" >> $DEPLOY_LOG
pkill -f 'gunicorn'
nohup /home/ubuntu/.local/bin/gunicorn -w 4 app:app -b 0.0.0.0:5000 > $APP_LOG 2> $ERROR_LOG &
disown


CURRENT_PID=$(pgrep -f 'gunicorn')
echo "$TIME_NOW > 실행된 프로세스 아이디 $CURRENT_PID 입니다." >> $DEPLOY_LOG


