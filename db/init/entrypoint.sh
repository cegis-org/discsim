#!/bin/bash
set -e

# Substitute environment variables in the SQL script
envsubst < /docker-entrypoint-initdb.d/init.sql > /docker-entrypoint-initdb.d/init_processed.sql

# Run the processed SQL script
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f /docker-entrypoint-initdb.d/init_processed.sql
