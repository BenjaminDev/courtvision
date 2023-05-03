docker pull heartexlabs/label-studio:latest
docker run --rm -it -p 8080:8080 -v /Users/benjamindecharmoy/projects/courtvision/labelstudiodata:/label-studio/data:rw heartexlabs/label-studio:latest
