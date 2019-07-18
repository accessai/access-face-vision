import base64

from sanic import Sanic
from sanic.response import json as sjson
from sanic_cors import CORS
from PIL import Image

from access_face_vision.exceptions import AccessException

app = Sanic(name='afv')


def setup_routes(cmd_args, afv, logger):

    CORS(app, resources={r"/*": {"origins": cmd_args.cors}}, automatic_options=True)

    def ae_error_handler(request, ae):
        logger.error("Error: {}".format(ae))
        return sjson({"Error": str(ae)}, status=ae.error_code)

    def general_error_handler(request, ex):
        logger.error("Error: {}".format(ex))
        return sjson({"Error": "Internal Server Error"}, status=500)

    app.error_handler.add(AccessException, ae_error_handler)
    app.error_handler.add(Exception, general_error_handler)

    def validate_request_keys(received_keys, required_keys):

        for key in required_keys:
            if key not in received_keys:
                raise AccessException('{} is required.'.format(key), error_code=400)

    async def handle_request(func, *args):
        resp_code = 200
        try:
            resp = func(*args)
        except AccessException as ae:
            logger.error("Error: {}".format(ae))
            resp_code = ae.error_code
            resp = {"Error": str(ae)}
        except Exception as ex:
            resp_code = 500
            logger.error("Error: {}".format(ex))
            resp = {'Error': 'Internal Server Error'}

        return sjson(resp, status=resp_code)

    @app.route('/', methods=['GET'])
    def status(request):
        return sjson({'Status': 'OK'})

    @app.route('/afv/v1/parse/image', methods=['POST'])
    async def parse_img(request):

        req_json = request.json

        received_keys = req_json.keys()
        required_keys = ['face_group']
        validate_request_keys(received_keys, required_keys)

        img_bytes = base64.decodebytes(req_json['image'].encode('utf-8'))
        img = Image.frombytes('RGB', (req_json['width'], req_json['height']), img_bytes, 'raw')
        return await handle_request(afv.parse_image, *(img, req_json['face_group']))

    @app.route("/afv/v1/facegroup", methods=['POST'])
    async def create_face_group(request):

        req_json = request.json

        received_keys = req_json.keys()
        required_keys = ['face_group']
        validate_request_keys(received_keys, required_keys)

        face_group = req_json['face_group']
        return await handle_request(afv.create_face_group, *(face_group,))

    @app.route("/afv/v1/facegroup", methods=['DELETE'])
    async def delete_face_group(request):

        req_json = request.json

        received_keys = req_json.keys()
        required_keys = ['face_group']
        validate_request_keys(received_keys, required_keys)

        face_group = req_json['face_group']
        return await handle_request(afv.delete_face_group, *(face_group,))

    @app.route("/afv/v1/facegroup/faceid", methods=['PUT'])
    async def append_to_face_group(request):

        req_json = request.json

        received_keys = req_json.keys()
        required_keys = ['face_group']
        validate_request_keys(received_keys, required_keys)

        if 'face_group' not in required_keys:
            raise AccessException('face_group is required', 400)

        face_group = req_json['face_group']
        label = req_json['label']
        img_bytes = base64.decodebytes(req_json['image'].encode('utf-8'))
        img = Image.frombytes('RGB', (req_json['width'], req_json['height']), img_bytes, 'raw')
        return await handle_request(afv.append_to_face_group, *(face_group, img, label))

    @app.route("/afv/v1/facegroup/faceids", methods=['POST'])
    async def get_face_ids(request):

        req_json = request.json

        received_keys = req_json.keys()
        required_keys = ['face_group']
        validate_request_keys(received_keys, required_keys)

        face_group = req_json['face_group']
        return await handle_request(afv.list_face_ids, *(face_group,))

    @app.route("/afv/v1/facegroup/faceid", methods=['DELETE'])
    async def delete_from_face_group(request):

        req_json = request.json

        received_keys = req_json.keys()
        required_keys = ['face_group', 'face_id']
        validate_request_keys(received_keys, required_keys)

        return await handle_request(afv.delete_from_face_group, *(req_json['face_group'], req_json['face_id']))
