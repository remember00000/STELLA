import os
from typing import List, Dict


class Const:
    # * path
    WORK_PATH: str = os.getcwd()  # * get the current working directory
    DATA_ROOT_PATH: str = os.path.join(WORK_PATH, "data")  # * data root path
    RESULT_ROOT_PATH: str = os.path.join(WORK_PATH, "result")  # * result root path
    # * task setting
    LIST_WISE: str = "list"
    PAIR_WISE: str = "pair"
    POINT_WISE: str = "point"
    # * argparser setting
    MODEL_LIST: List[str] = ["text-davinci-001", "text-davinci-002", "text-davinci-003", "gpt-3.5-turbo"]
    MODEL_DEFAULT: str = "gpt-3.5-turbo"
    DOMAIN_LIST: List[str] = ["Book",  "Music", "News"]
    DOMAIN_DEFAULT: str = "Book"
    TASK_LIST: List[str] = [LIST_WISE, PAIR_WISE, POINT_WISE]
    TASK_DEFAULT: str = LIST_WISE
    # * data infomation
    DATA_COL_NAME_LIST: List[str] = ["userid", "itemid_history", "pos_target_index", "itemid_candidate","pos_id_target"]##pos_id_target  "pair_pos_target_index", "pair_itemid_candidate",
    DATA_EVAL_COL_NAME_LIST: List[str] = ["itemid_history", "itemid_candidate"]
    # * model setting
    COMPLETION_MODEL_LIST: List[str] = ["text-davinci-001", "text-davinci-002", "text-davinci-003"]
    CHAT_MODEL_LIST: List[str] = ["gpt-3.5-turbo"]
    URL_DICT: Dict[str, str] = {
        "gpt-3.5-turbo": ,
        "text-davinci-003": ,
        "text-davinci-002": ,
        "text-davinci-001": ,
    TOKENIZER_DICT: Dict[str, str] = {
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "text-davinci-001": "r50k_base",
    }
