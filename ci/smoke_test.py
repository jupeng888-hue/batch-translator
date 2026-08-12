# -*- coding: utf-8 -*-
"""CI 冒烟测试：中文路径文件夹下 图片(en,ja) + 视频(en) 全链路。"""
import os, shutil, sys

sys.path.insert(0, os.path.abspath('translator_app'))
from core.batch import run_batch

stage = os.path.join('ci', '中文测试输入')
shutil.rmtree(stage, ignore_errors=True)
os.makedirs(stage)
shutil.copy(os.path.join('ci', 'media', 'sample1.jpg'), stage)
shutil.copy(os.path.join('ci', 'media', 'sample.mp4'), stage)

out = os.path.join('ci', '输出结果')
shutil.rmtree(out, ignore_errors=True)

r = run_batch(stage, out, ['en', 'ja'], do_images=True, do_videos=True)
print('RESULT:', r)
assert r['failed'] == 0, f"有失败任务: {r}"
assert r['done'] == 4, f"任务数不符: {r}"

expect = []
for lang in ['英语', '日语']:
    expect.append(os.path.join(out, lang, 'sample1.jpg'))
expect.append(os.path.join(out, '英语', 'sample.mp4'))
srt = os.path.join(out, '英语', 'sample.srt')
for p in expect:
    assert os.path.exists(p) and os.path.getsize(p) > 1000, f"输出缺失或过小: {p}"
    print('OK', p, os.path.getsize(p))
if os.path.exists(srt):
    print('OK(可选)', srt, os.path.getsize(srt))
else:
    print('INFO 画面文字已原位替换，未生成外挂字幕(符合预期):', srt)
print('SMOKE TEST PASSED')
