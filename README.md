## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `Mask_RCNN-master/requirements.txt`.


## Pretreatment

**!!! ATTENTION !!! ATTENTION !!! ATTENTION !!!**

The execution sequence of the preprocessing code should be `extract_video_frames_separate.py → gait_synthesis_visualization.py → pretreatmenr_rotate.py`

The obtained RGB video cannot be directly used for model learning, it needs to be converted into a binary image first. To be able to completely extract the human body contour, we adopt the detection method of the `Mask R-CNN` model. First, we configure and load the pre-trained `Mask R-CNN` model, which can be used to detect human body contour targets. Secondly, we perform frame-by-frame detection on the obtained RGB video and use masks to record the pixel positions of the contour in the image. Finally, based on the covered areas of the masks, we create a binary image of the same size to obtain the initial contour map.

For detailed instructions of `Mask R-CNN`, please refer to the `README.md` in the folder `Mask_RCNN-master`.

`extract_video_frames_separate.py` in `Mask R-CNN` extract the basic silhouette frames from the RGB video, clean up the background and noise, and form a standard three-level directory structure. `extract_video_frames_separate.py` solve the reliable extraction from the original RGB video to the silhouette.

Pretreatment your dataset by
```
python extract_video_frames_separate.py --VIDEO_PATH = ".\Mask_RCNN-master\video.mp4"
```
- `--VIDEO_PATH` **(NECESSARY)** Root path for input video

`gait_synthesis_visualization.py.py` perform mask decomposition and reconstruction on the silhouette, synthesize the enhanced-edge silhouette result, and save the intermediate visualization for inspection.

Pretreatment your dataset by
```
python gait_synthesis_visualization.py --input_root=".\output\pretreatment" --output_root=".\output\output_synthesis" --visual_output_root=".\output\visualization"
```
- `--input_root` **(NECESSARY)** Root path that use the `--output_root` directory by `video.py` as the root directory of the three-layer structure for input.
- `--output_root` **(NECESSARY)** Root path for saving the synthesized silhouette with the same structure form as the `--input_root`. E.g. `input_root\001\seq1\090\frame_00012.png` → `output_root\001\seq1\090\frame_00012.png`
- `--visual_output_root` **(NECESSARY)** Root path for save the intermediate visual results.

`pretreatment_rotate.py` perform geometric alignment and orientation correction on the synthesized silhouette, and unify it to 64×64 to ensure the spatio-temporal consistency of the input model.

Pretreatment your dataset by
```
python pretreatmenr_rotate.py --input_path=".\output\output_synthesis" --output_path=".\output\output_synthesis_rotate" --worker_num=1 --log=TRUE --debug=FALSE
```
- `--input_path` **(NECESSARY)** Root path of raw video.
- `--output_path` **(NECESSARY)** Root path for video output.
- `--log_file` Log file path. #Default: '.\ pretreatment.log'
- `--log` If set as True, the aligned call attempts will be saved. 
Otherwise, only warnings and errors will be saved. #Default: False
- `--worker_num` How many subprocesses to use for data pretreatment. Default: 1
- `--debug` If set as True, all logs will be saved. 
Otherwise, the operation will not be executed. #Default: False
