@echo off
set PORT=9099
set HOST=127.0.0.1

call conda activate pipelines

uvicorn main:app --host %HOST% --port %PORT% --forwarded-allow-ips '*'