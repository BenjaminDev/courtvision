#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../courtvision"
poetry build
cp ../dist/*.whl ../courtvision_balldetector/
cd ../courtvision_balldetector
# sudo chown -R :0 data # fix: PermissionError: [Errno 13] Permission denied: '/label-studio/data/media'
# sudo chmod a+w /mnt/vol_b/labeldatastudio # fix: PermissionError: [Errno 13] Permission denied: '/label-studio/data/media'
docker-compose build
docker-compose up
