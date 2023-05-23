#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../courtvision"
poetry build
cp ../dist/*.whl ../courtvision_balldetector/
cd ../courtvision_balldetector
docker-compose build
docker-compose up
