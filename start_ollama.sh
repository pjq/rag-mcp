#!/bin/bash

MODEL=${1:-llama3}
PORT=11434

echo "🔍 Checking if Ollama is running on port $PORT..."

if lsof -i tcp:$PORT >/dev/null 2>&1; then
  echo "✅ Ollama is already running on port $PORT."
else
  echo "🚀 Starting Ollama with model: $MODEL ..."
  nohup ollama run "$MODEL" > ollama.log 2>&1 &
  sleep 2
  echo "⏳ Waiting for Ollama to be ready..."

  # Optionally poll until Ollama is up
  until curl -s http://localhost:$PORT > /dev/null; do
    sleep 1
  done

  echo "✅ Ollama is now running."
fi
