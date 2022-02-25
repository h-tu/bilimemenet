# CS685 Spring 2022 
# Feb. 24, 2022
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
from bv2av import bv_to_av

from collections import Counter
from bilibili_api import video, comment, sync, exceptions

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
    
    tmp1_lst, tmp2_lst = [], []
    for video_id in (bvid_lst):
        title, tid, channel_id, view_count, bc_lst, cmt_lst = asyncio.run(process_single_video(video_id))
        for bc_info in bc_lst:
            bc, freq = bc_info
            tmp = [bc, freq, video_id, title, tid, channel_id, view_count]
            tmp1_lst.append(tmp)
        for cmt_info in cmt_lst:
            cmt, freq = cmt_info
            tmp = [cmt, freq, video_id, title, tid, channel_id, view_count]
            tmp2_lst.append(tmp)
    return tmp1_lst, tmp2_lst


def clean_text(text):
    text = ''.join(char for char in text if char.isalnum())
    text = re.sub(misc_symbols,'', text)
    text = re.sub(symbols_punct,'', text)
    text = re.findall(r'[\u4e00-\u9fffa-zA-Z0-9]+', text)
    text = ''.join(text)
    return None if text == '' else text


def process_text_lst(txt_lst, dm_or_cmt = True):
    text_lst = []
    for i in txt_lst:
        text = clean_text(i.text if dm_or_cmt else i) 
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
    dms, info, cmt_lst = None, None, []
    try:
        dms = sync(v.get_danmakus(0))
    except exceptions.ResponseCodeException:
        pass
    try:
        info = await v.get_info()
    except exceptions.ResponseCodeException:
        pass

    try:
        cmt_lst = sync(get_comment(BVID))
    except exceptions.NetworkException:
        pass

    if dms is not None and info is not None:
        title = clean_text(info['title'])
        channel_id = info['owner']['name']
        tid = info['tid'] 
        view_count = info['stat']['view']
        bc_lst = process_text_lst(dms)
        cmt_lst = process_text_lst(cmt_lst, False)
        return (title, tid, channel_id, view_count, bc_lst, cmt_lst)
    else:
        return (None, None, None, None, [], [])


def list_to_csv(lst, name):
    df = pd.DataFrame(lst, columns = [name, 'Frequency', 'BVID', 'Source Video Title', 'Category ID', 'Channel ID', 'Source Video View Count'])
    df = df.drop_duplicates(subset=['BVID', name])
    df = df.sort_values(by=['Frequency'])
    df.to_csv('{}.csv'.format(name)) 
    return df


async def get_comment(BVID):
    comment_lst = []
    comments, page, count = [], 1, 0
    AVID = bv_to_av(BVID)

    if AVID != '获取av号失败':
        AVID = int(AVID)
        while True:
            c = await comment.get_comments(AVID, comment.ResourceType.VIDEO, page)      # 获取评论
            comments.extend(c['replies'])                                               # 存储评论
            count += c['page']['size']                                                  # 增加已获取数量
            page += 1                                                                   # 增加页码

            if count >= c['page']['count']:
                break                                                                   # 当前已获取数量已达到评论总数，跳出循环

        for cmt in comments:
            comment_lst.append(cmt['content']['message'])
            # print(f"{cmt['member']['uname']}: {cmt['content']['message']}")

        # 打印评论总数
        # print(f"\n\n共有 {count} 条评论（不含子评论）")
    return comment_lst