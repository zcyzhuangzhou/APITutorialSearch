import os

"""
this file contain some util method for file

"""
# todo: complete this method
class DirUtil:
    def __init__(self):
        pass

    @staticmethod
    def create_file_dir(dir_path):
        is_exits = os.path.exists(dir_path)
        if not is_exits:
            os.makedirs(dir_path)
            print("the output dir %r is not exist, creating" % dir_path)
        else:
            print("the output dir %r is existing, not creating" % dir_path)
