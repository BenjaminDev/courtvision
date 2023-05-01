podman pull heartexlabs/label-studio:latest
podman run --rm -it -p 8080:8080 -v /mnt/Users/benjamindecharmoy/projects/courtvision/labelstudiodata:/label-studio/data:rw heartexlabs/label-studio:latest
