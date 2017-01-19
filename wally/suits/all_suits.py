from .io.fio import FioTest
# from .suits.itest import TestSuiteConfig
# from .suits.mysql import MysqlTest
# from .suits.omgbench import OmgTest
# from .suits.postgres import PgBenchTest


all_suits = {suite.name: suite for suite in [FioTest]}
