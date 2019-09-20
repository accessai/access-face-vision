[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Access Face Vision

Face detection and recognition Application.

### With pip
 - ##### Installation
```bash
pip install access-face-vision
```

- ##### Training/Creating FaceGroup
```bash
python -m access_face_vision --mode train --img_dir samples/celebrities --face_group celebrities

# Directory structure
# **/Images/
#          A/
#           A_K_01.jpg
#           A_K_02.jpg
#          B/
#           B_S_01.jpg
#           B_S_02.jpg
```

- ##### Running Inferences
```bash
# Live video feed
python -m access_face_vision --mode live-video --camera-index 0 --camera_wait 30 --face_group celebrities

# server mode
python -m access_face_vision --mode server --face_group_dir ./
```
Use [access-client](https://github.com/accessai/access-client) to make requests to this server


### Docker image

- Docker Image build
```bash
docker build -t access_face_vision:latest .
```

 - Docker run
 ```bash
# we will use it as root directory for access_face_vision application
mkdir -p accessai/afv

# Start server
docker run -v $(pwd)/afv:/accessai/afv python -m access_face_vision --mode server

# Start camera feed processor
docker run -v $(pwd)/afv:/accessai/af v python -m access_face_vision --mode server
```

## Contribution
 Contributions are welcome. Feel free to raise PRs! with any improvements.


## Credit
Face Encoder model: https://github.com/nyoki-mtl/keras-facenet