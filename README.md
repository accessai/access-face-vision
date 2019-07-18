# Access Face Vision

Face detection and recognition APIs.

### With pip
 - ##### Installation
```bash
pip install access-face-vision
```

- ##### Running Inferences
```bash
# server mode
python -m access_face_vision --mode server

curl -X POST \
  http://localhost:5001/afv/v1/parse \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -d '{
	"face_group": "default",
	"width": 800,
	"height": 600,
	"image": "base64-encoded"}'


# Live video feed
python -m access_face_vision --mode live-video --camera-index 0 --camera_wait 25
python -m access_face_vision --mode live-video --camera-index rtp://camera-url


# Pre recorded video
python -m access_face_vision --mode recorded-video --video_dir path-to-video-directory 
python -m access_face_vision --mode recorded-video --video_path path-to-video-file 
```

- ##### Training face embeddings
```bash
python -m access_face_vision.train_face_recognition_model --mode training --img_dir path-to-image-directory

# Directory structure
# **/Images/
#          A/
#           A_K_01.jpg
#           A_K_02.jpg
#          B/
#           B_S_01.jpg
#           B_S_02.jpg

```


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
docker run -v $(pwd)/afv:/accessai/afv python -m access_face_vision --mode server
```

