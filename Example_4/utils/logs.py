# @Time    : 2019/4/11 19:15
# @Email  : wangchengo@126.com
# @File   : logs.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import logging


class Logger(object):

    def __init__(self, log_file_name, log_level, logger_name):
        # 创建一个logger
        self.__logger = logging.getLogger(logger_name)

        # 指定日志的最低输出级别，默认为WARN级别
        self.__logger.setLevel(log_level)

        # 创建一个handler用于写入日志文件
        file_handler = logging.FileHandler(log_file_name)

        # 创建一个handler用于输出控制台
        console_handler = logging.StreamHandler()

        # 定义handler的输出格式
        console_formatter = logging.Formatter('- %(levelname)s: %(message)s')
        file_formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # 给logger添加handler
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger
