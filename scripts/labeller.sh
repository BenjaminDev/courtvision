podman pull heartexlabs/label-studio:latest
podman run --rm -it -p 8080:8080 -v $(pwd)/labelstudiodata:/label-studio/data heartexlabs/label-studio:latest
