sudo apt-get install p7zip-full
export PYTHONPATH=src:$PYTHONPATH
echo 'export PYTHONPATH=src:$PYTHONPATH' >> ~/.bashrc 


if [ -d "data/" ]; then
    echo "The folder 'data/' exists."
else
    git clone https://oauth2:cVnpw359B8V87JdxmFsS@www.modelscope.cn/datasets/monster119120/UI_grounding_CN.git
    mv UI_grounding_CN/ data/
fi


cd data/
if [ -d "images/" ]; then
    echo "The folder 'images/' exists."
else
    tar -xzvf images.tar.gz
fi
cd ..

pip3 install -r requirements.txt
export MODEL_NAME="Molmo-7B-D-0924"
modelscope download --model "LLM-Research/Molmo-7B-D-0924" --local_dir $MODEL_NAME