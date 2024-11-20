#! /bin/bash

# Langchain
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY=$(cat API_Key/Langchain_key.txt)
export LANGCHAIN_PROJECT="RAG_ROG"  
