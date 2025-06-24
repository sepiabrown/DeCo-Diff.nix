:loop
set NO_ALBUMENTATIONS_UPDATE=1
py -3.11 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 project/train_DeCo_Diff.py --input-json input_json/train_input.json
goto loop