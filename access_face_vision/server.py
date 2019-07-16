import logging
import base64
import asyncio
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger('SERVER')

from sanic import Sanic
from sanic.response import json as sjson
from sanic_cors import CORS
import numpy as np
from PIL import Image

from access_face_vision.exceptions import AccessException

app = Sanic(name='afv')


def setup_routes(cmd_args, afv):

    CORS(app, resources={r"/*": {"origins": cmd_args.cors}})

    # def check_cors(f):
    #     def cors(*args, **kwargs):
    #         request = args[0]
    #
    #         request.setHeader('Access-Control-Allow-Origin', cmd_args.cors)
    #         request.setHeader('Access-Control-Allow-Methods', 'POST')
    #         request.setHeader('Access-Control-Allow-Headers', '*')
    #         if request.method.decode('utf-8', 'strict') == 'OPTIONS':
    #             return ''
    #
    #         return f(*args, **kwargs)
    #
    #     return cors

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

    @app.route('/', methods=['GET', 'OPTIONS'])
    def status(request):
        return "{'Status': 'OK'}"

    @app.route('/afv/v1/parse/image', methods=['POST', 'OPTIONS'])
    async def parse_img(request):

        req_json = request.json
        img_bytes = base64.decodebytes(req_json['image'].encode('utf-8'))
        img = Image.frombytes('RGB', (req_json['width'], req_json['height']), img_bytes, 'raw')

        return await handle_request(afv.parse_image, *(img, req_json['face_group']))

    @app.route("/afv/v1/facegroup", methods=['POST', 'OPTIONS'])
    async def create_face_group(request):

        try:
            request_json = request.json
        except:
            request.setResponseCode(500)
            return sjson({"Error":"Bad Request"}, status=500)


        return handle_request(request, afv.create_face_group, *(request_json['face_group'],))

    @app.route("/afv/v1/facegroup", methods=['PUT'])
    async def append_to_face_group(request):

        try:
            request_json = request.json
            img = np.fromstring(base64.decodebytes(request_json['image'].encode('utf-8')), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except Exception as ex:
            logger.error(ex)
            request.setResponseCode(500)
            return sjson({"Error":"Bad Request"}, status=500)

        return handle_request(request, afv.append_to_face_group, *(request_json['face_group'],
                                                                   img,
                                                                   request_json['label']))


if __name__ == '__main__':
    pass
