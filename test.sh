


python3 deploy.py \
 	--model ./pre_trained/erd_seg_matting/model/model_obj.pth \
	--inputPath ./test/source \
	--savePath ./test/source_segment \
	--size=256 \
	--without_gpu