import argparse
import glob
import multiprocessing as mp
import os
import time
from torch.cuda.amp import autocast
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from minvis import add_minvis_config
from uni_dvps import add_uni_dvps_config
from predictor import VisualizationDemo_Panoptic

from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')

def setup_cfg(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	add_deeplab_config(cfg)
	add_maskformer2_config(cfg)
	add_maskformer2_video_config(cfg)
	add_minvis_config(cfg)
	add_uni_dvps_config(cfg)
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.freeze()
	return cfg

def get_parser():
	parser = argparse.ArgumentParser(description="unidvps demo for builtin configs")
	parser.add_argument(
		"--config-file",
		default="configs/CityscapesDVPS/R50.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--input",
		help="directory of input video frames",
		required=True,
	)
	parser.add_argument(
		"--output",
		help="directory to save output frames",
		required=True,
	)
	parser.add_argument(
		"--confidence-threshold",
		type=float,
		default=0.5,
		help="Minimum score for instance predictions to be shown",
	)
	parser.add_argument(
		"--opts",
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=[],
		nargs=argparse.REMAINDER,
	)
	return parser

if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	args = get_parser().parse_args()
	setup_logger(name="fvcore")
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	cfg = setup_cfg(args)
	demo = VisualizationDemo_Panoptic(cfg)

	assert args.input and args.output

	video_root = args.input
	output_root = args.output

	os.makedirs(output_root, exist_ok=True)
	
	frames_path = video_root
	frames_path = glob.glob(os.path.expanduser(os.path.join(frames_path, '*.png')))
	frames_path.sort()

	vid_frames = []
	for path in frames_path:
		img = read_image(path, format="RGB")
		vid_frames.append(img)

	start_time = time.time()
	with autocast():
		predictions, visualized_output, visualized_depth_output = demo.run_on_video(vid_frames)

	# save frames
	for path, _vis_output in zip(frames_path, tqdm(visualized_output, initial=1)):
		out_filename = os.path.join(output_root, os.path.basename(path).split('.png')[0]+'_seg.png')
		_vis_output.save(out_filename)
	print("Panoptic segmentation results are saved in {}".format(output_root))

	for path, _vis_output in zip(frames_path, tqdm(visualized_depth_output, initial=1)):
		out_filename = os.path.join(output_root, os.path.basename(path).split('.png')[0]+'_depth.png')
		_vis_output.save(out_filename)
	print("Depth estimation results are saved in {}".format(output_root))

