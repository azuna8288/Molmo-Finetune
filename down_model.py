#验证SDK token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('0904907c-fc44-4147-af35-fee543d9f27a')

import argparse
import os

parser = argparse.ArgumentParser(description="test")
parser.add_argument('--model_version', default="Molmo7b_CN_v8", type=str, help='model path')
args = parser.parse_args()

#模型下载
from modelscope import snapshot_download
if not os.path.exists(args.model_version):
    model_dir = snapshot_download(f'monster119120/{args.model_version}', local_dir=args.model_version, max_workers=16)