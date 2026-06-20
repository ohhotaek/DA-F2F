from detectron2.data.datasets import register_coco_instances

# Cityscapes 
register_coco_instances("cityscapes_train", {},         "/mnt/hdddata/hotaek/mrt/cityscapes_train_cocostyle.json",                  "/mnt/hdddata/hotaek/dataset/cityscapes/leftImg8bit/train/")
register_coco_instances("cityscapes_val",   {},         "/mnt/hdddata/hotaek/mrt/cityscapes_val_cocostyle.json",                    "/mnt/hdddata/hotaek/dataset/cityscapes/leftImg8bit/val/")

# Foggy Cityscapes
register_coco_instances("cityscapes_foggy_train", {},   "/mnt/hdddata/hotaek/mrt/foggy_cityscapes_train_cocostyle.json",   "/mnt/hdddata/hotaek/dataset/leftImg8bit_foggy/train/")
register_coco_instances("cityscapes_foggy_val", {},     "/mnt/hdddata/hotaek/mrt/foggy_cityscapes_val_cocostyle.json",     "/mnt/hdddata/hotaek/dataset/leftImg8bit_foggy/val/")

# Sim10k and cityscapes_cars
register_coco_instances("sim10k_cars_train", {},             "/mnt/hdddata/hotaek/mrt/sim10k_train_cocostyle.json",                  "/mnt/hdddata/hotaek/VOC2012/JPEGImages/")
register_coco_instances("cityscapes_cars_val",   {},         "/mnt/hdddata/hotaek/mrt/cityscapes_val_caronly_cocostyle.json",                    "/mnt/hdddata/hotaek/dataset/cityscapes/leftImg8bit/val/")

# BDD100k
register_coco_instances("bdd100k_train", {},   "/mnt/hdddata/hotaek/mrt/bdd100k_daytime_train_cocostyle.json",   "/mnt/hdddata/hotaek/dataset/bdd100k/images/100k/train/")
register_coco_instances("bdd100k_val", {},     "/mnt/hdddata/hotaek/mrt/bdd100k_daytime_val_cocostyle.json",     "/mnt/hdddata/hotaek/dataset/bdd100k/images/100k/val/")