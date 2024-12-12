from modelscope.hub.constants import ModelVisibility
from modelscope.hub.api import HubApi
import os

YOUR_ACCESS_TOKEN = '0904907c-fc44-4147-af35-fee543d9f27a'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)


if os.environ.get("MODEL_NAME", None) == None:
    print('no such ckpt, please check')
    exit(0)


username = 'monster119120'
version = 8
create_sucess = False
while not create_sucess:
    try:
        model_name = f'Molmo7b_CN_v{version}'
        model_id=f'{username}/{model_name}'

        api.create_model(
            model_id,
            visibility=ModelVisibility.PRIVATE,
        )
        create_sucess = True
    except:
        print('try raise version')
        version += 1

api.push_model(
    model_id=model_id, 
    model_dir=os.environ.get("MODEL_NAME", None)
)