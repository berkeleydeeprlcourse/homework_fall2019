class BaseModel(object):

    def __init__(self, **kwargs):
        pass

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        raise NotImplementedError

    def get_prediction(self, ob_no, ac_na):
        raise NotImplementedError