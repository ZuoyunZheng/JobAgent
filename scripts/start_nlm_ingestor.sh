#!/usr/bin/sh
docker pull ghcr.io/nlmatics/nlm-ingestor:latest
docker run -p 5010:5001 --name nlm-ingestor ghcr.io/nlmatics/nlm-ingestor:latest
