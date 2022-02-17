from bilibili_api import video, sync
import time
import datetime
import re

v = video.Video(bvid='BV1B3411L7Y6')

dms = sync(v.get_danmakus(0))
file_name = 'danmu_file_' + str(datetime.datetime.now().strftime("%H_%M_%S"))
print(file_name)
f = open("DanMu/"+file_name + ".txt", "a")

hiragana_full = r'[ぁ-ゟ]'
katakana_full = r'[゠-ヿ]'
kanji = r'[㐀-䶵一-鿋豈-頻]'
radicals = r'[⺀-⿕]'
katakana_half_width = r'[｟-ﾟ]'
alphanum_full = r'[！-～]'
symbols_punct = r'[、-〿]'
misc_symbols = r'[ㇰ-ㇿ㈠-㉃㊀-㋾㌀-㍿]'
ascii_char = r'[ -~]'

for dm in dms:
    text = dm.text 
    text = ''.join(char for char in text if char.isalnum())
    text = re.sub(misc_symbols,'', text)
    text = re.sub(symbols_punct,'', text)
    text = re.findall(r'[\u4e00-\u9fff]+', text)
    text = ''.join(text)
    print(text)
    if text == '':
        continue
    f.write(text + ",")
    f.write(str(datetime.datetime.fromtimestamp(dm.send_time)))
    f.write("\n")
    #print(dm.dm_time)
    #print(datetime.datetime.fromtimestamp(dm.send_time))
f.close()