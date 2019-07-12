import os
import logging

env=os.getenv('ENV', None)

logger = logging.getLogger('SERVER')


import json
from klein import Klein

from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks, returnValue

app = Klein()
reactor.suggestThreadPoolSize(5)


def setup_routes(args, afv):

    @app.route('/', methods=["GET"])
    def status(request):
        return "{'Status': 'OK'}"

    @app.route('/parse/image', methods=["POST", 'OPTIONS'])
    @inlineCallbacks
    def main(request):
        request.setHeader('Content-Type', 'application/json')
        request.setHeader('Access-Control-Allow-Origin', args.cors)
        request.setHeader('Access-Control-Allow-Methods', 'POST')
        request.setHeader('Access-Control-Allow-Headers', '*')
        if request.method.decode('utf-8', 'strict') == 'OPTIONS':
            return ''

        try:
            img_bytes = request.content.file.raw
            request.setHeader('Content-Type', 'application/json')
            resp = yield threads.deferToThread(afv.parse_image, img_bytes)
            resp = json.dumps(resp, )
        except Exception as ex:
            logger.error("Error: {}".format(ex))
            request.setResponseCode(500)
            return json.dumps({'Error': 'Internal Server Error'})

        return resp

    def validate_request_body(f):

        def validate(*args, **kwargs):
            request = args[0]
            request_json = args[1]

            if not (request_json.get('item_id')):
                request.setHeader('Content-Type', 'application/json')
                request.setResponseCode(400)
                return json.dumps({'Error': 'Bad request body'})

            return f(*args, **kwargs)

        return validate

    @validate_request_body
    def serve(request, request_json):
        request.setHeader('Content-Type', 'application/json')

        logger.info("Request body: {}".format(request_json))
        resp, resp_code = afv.generate_recommendations(**request_json)

        request.setResponseCode(resp_code)
        logger.info("Response body: {}".format(resp))
        returnValue(json.dumps(resp))


resource = app.resource


if __name__ == '__main__':
    pass
