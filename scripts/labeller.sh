docker pull heartexlabs/label-studio:latest
docker run -it -p 8080:8080 -v $(pwd)/labelstudiodata:/label-studio/data heartexlabs/label-studio:latest
