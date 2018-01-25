import os
import time

def init_log(category,name):
    """
    Create the learning log
    """
    now_time = time.strftime("%Y-%m-%d", time.localtime())

    if not os.path.exists("../Log/{0}".format(category)):
        os.mkdir("../Log/{0}".format(category))
    if not os.path.exists("../Log/{0}_{1}.txt".format(now_time, name)):
        f = open("../Log/%s_QBrainSimply_log.txt" % now_time, 'r')
        f.close()

def record():
    pass # TODO

