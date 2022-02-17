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

text = '初めての駅 自由が丘の駅で、大井町線から降りると、ママは、トットちゃんの手を引っ張って、改札口を出ようとした。ぁゟ゠ヿ㐀䶵一鿋豈頻⺀⿕｟ﾟabc！～、〿ㇰㇿ㈠㉃㊀㋾㌀㍿'

print('Original text string:', text, '\n')
print('All kanji removed:', remove_unicode_block(kanji, text))
print('All hiragana in text:', ''.join(extract_unicode_block(hiragana_full, text)))

# for dm in dms:
#     text = dm.text 
#     text = ''.join(char for char in text if char.isalnum())
#     print(text)
#     f.write(text + ",")
#     f.write(str(datetime.datetime.fromtimestamp(dm.send_time)))
#     f.write("\n")
#     print(dm.dm_time)
#     print(datetime.datetime.fromtimestamp(dm.send_time))
f.close()