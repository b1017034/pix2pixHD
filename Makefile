.PHONY: test

NAME = test

test:
	python test.py --name ${NAME}  --gpu_ids 0 --dataroot ./datasets/${NAME}/ --label_nc 0 --loadSize 1024 --no_instance --resize_or_crop none
	
train:
	python train.py --name ${NAME}  --gpu_ids 0 --dataroot ./datasets/${NAME}/ --label_nc 0 --loadSize 1024 --data_type 8 --no_instance --resize_or_crop none

export:
	python test.py --name ${NAME} --dataroot ./datasets/${NAME}/ --label_nc 0 --loadSize 1920 --which_epoch latest --no_instance --export_onnx ${NAME}.onnx
	
export-onnx:
	python export_onnx.py --name ${NAME} --dataroot ./datasets/${NAME}/ --label_nc 0 --loadSize 1920 --which_epoch latest --no_instance --export_onnx ${NAME}.onnx
