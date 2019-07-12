# Access Face Vision

Face detection and recognition APIs.

### Docker image

- Docker Image build
```bash
docker build -t access_face_vision:latest .
```

 - Docker run
 ```bash
# we will use it as root directory for access_face_vision application
mkdir accessai

# Start server
docker run -v $(pwd)/accessai:/accessai python -m access_face_vision

# Start camera feed processor
TODO
```

### Inference
 - Image
 ```bash
    curl -X POST \
  http://localhost:5001/parse \
  -F data=@"path to jpeg image"
```

