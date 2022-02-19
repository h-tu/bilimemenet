# CS685 Spring 2022 
# Feb. 19, 2022
# Hongyu Tu

import re
import time
import asyncio
import datetime
import requests
import numpy as np
import pandas as pd 
import nest_asyncio
from tqdm import tqdm

from collections import Counter
from bilibili_api import video, sync, exceptions

hiragana_full = r'[ぁ-ゟ]'
katakana_full = r'[゠-ヿ]'
kanji = r'[㐀-䶵一-鿋豈-頻]'
radicals = r'[⺀-⿕]'
katakana_half_width = r'[｟-ﾟ]'
alphanum_full = r'[！-～]'
symbols_punct = r'[、-〿]'
misc_symbols = r'[ㇰ-ㇿ㈠-㉃㊀-㋾㌀-㍿]'
ascii_char = r'[ -~]'

def process_category(rid = 47, day = 7):
    # rid: category ID, day: 3：3 day ranking、7：week ranking
    api = "https://api.bilibili.com/x/web-interface/ranking/region?rid={}&day={}".format(rid, day)
    bvid_lst = requests.get(api).text.split("\"bvid\":\"")[1:]
    bvid_lst = [line.split('\"')[0] for line in bvid_lst]
    
    tmp_lst = []
    for video_id in (bvid_lst):
        title, tid, channel_id, view_count, bc_lst = asyncio.run(process_single_video(video_id))
        for bc_info in bc_lst:
            bc, freq = bc_info
            tmp = [bc, freq, video_id, title, tid, channel_id, view_count]
            tmp_lst.append(tmp)
    return tmp_lst


def clean_text(text):
    text = ''.join(char for char in text if char.isalnum())
    text = re.sub(misc_symbols,'', text)
    text = re.sub(symbols_punct,'', text)
    text = re.findall(r'[\u4e00-\u9fffa-zA-Z0-9]+', text)
    text = ''.join(text)
    return None if text == '' else text


def process_dms(dms):
    text_lst = []
    for dm in dms:
        text = clean_text(dm.text) 
        if text == None:
            continue
        text_lst.append(text)
        # date = str(datetime.datetime.fromtimestamp(dm.send_time))
        # time = str(dm.dm_time)
    
    output_lst = []
    tmp_dict = dict(Counter(text_lst))
    for i in tmp_dict.keys():
        output_lst.append((i, tmp_dict[i]))
    return output_lst


def init_category_dic():
    f = open('category_info.txt', encoding="utf8")
    # File format: Chinese category name, English category name, Category ID, Parent Category ID
    content = f.readline()
    cat_dic = {}
    while content:
        c_name, e_name, cid, parent_id = content.replace(" ", "").replace("\n", "").split(',')
        cat_dic[int(cid)] = (c_name, e_name,  int(parent_id))
        content = f.readline()
    f.close()
    return cat_dic


async def process_single_video(BVID):
    v = video.Video(bvid=BVID)
    dms, info = None, None
    try:
        dms = sync(v.get_danmakus(0))
    except exceptions.ResponseCodeException:
        pass
    try:
        info = await v.get_info()
    except exceptions.ResponseCodeException:
        pass

    if dms is not None and info is not None:
        title = clean_text(info['title'])
        channel_id = info['owner']['name']
        tid = info['tid'] 
        view_count = info['stat']['view']
        bc_lst = process_dms(dms)
        return (title, tid, channel_id, view_count, bc_lst)
    else:
        return (None, None, None, None, [])

def list_to_csv(lst):
    df = pd.DataFrame(lst, columns = ['Bullet Chat', 'Frequency', 'BVID', 'Source Video Title', 'Category ID', 'Channel ID', 'Source Video View Count'])
    df.to_csv('out.csv') 
    return df