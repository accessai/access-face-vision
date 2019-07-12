class AccessException(Exception):

    def __init__(self, *args, **kwargs):
        self.message = args[0]
        self.error_code = kwargs.get('error_code') or 500

    def __str__(self):
        return self.message