REQUIRED_FPS = 3

def clean_queue(queue):

    try:
        while not queue.empty():
            queue.get(block=False)
    except Exception as ex:
        pass
    finally:
        return True
