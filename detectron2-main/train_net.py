# from detectron2.engine import DefaultTrainer
from CustomTrainer import CustomTrainer
from detectron2.config import get_cfg
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator #, inference_on_dataset
# from detectron2.data import build_detection_test_loader
import shutil

register_coco_instances("train", {}, "./datasets/coco/annotations/instances_train2017.json", "./datasets/coco/train_2017/")
register_coco_instances("test", {}, "./datasets/coco/annotations/instances_test2017.json", "./datasets/coco/test_2017/")

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("test",)
cfg.DATALOADER.NUM_WORKERS = 16
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = (2500)
cfg.SOLVER.CHECKPOINT_PERIOD = 100
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.NUM_GPUS = 4
cfg.CUDA_VISIBLE_DEVICES = 0,1,2,3
cfg.TEST.EVAL_PERIOD = 100
cfg.INPUT.RANDOM_FLIP = "horizontal"

# delete old outputs
shutil.rmtree(cfg.OUTPUT_DIR)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
open('./output/metrics.json', 'w').close()

trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# evaluator = COCOEvaluator("test", cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, "test")
# print(inference_on_dataset(trainer.model, val_loader, evaluator))

trainer.build_evaluator(cfg, "test")
test_datasets = ["test"]
evaluator = [COCOEvaluator(test_set, cfg, False) for test_set in test_datasets]
metrics = CustomTrainer.test(cfg, trainer.model, evaluator)