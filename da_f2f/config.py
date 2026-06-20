from detectron2.config import CfgNode as CN


def add_da_f2f_config(cfg):
    _C = cfg

    # Datasets and sampling
    _C.DATASETS.UNLABELED = tuple()
    _C.DATASETS.BATCH_COMPONENTS = ("labeled_weak", )
    _C.DATASETS.BATCH_RATIOS = (1,)
    _C.DATASETS.BLEND = False

    # Strong augmentations
    _C.AUG = CN()
    _C.AUG.WEAK_INCLUDES_MULTISCALE = True
    _C.AUG.LABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.UNLABELED_INCLUDE_RANDOM_ERASING = True
    _C.AUG.LABELED_MIC_AUG = False
    _C.AUG.UNLABELED_MIC_AUG = False
    _C.AUG.MIC_RATIO = 0.5
    _C.AUG.MIC_BLOCK_SIZE = 32

    # EMA of student weights
    _C.EMA = CN()
    _C.EMA.ENABLED = False
    _C.EMA.ALPHA = 0.9996
    _C.EMA.LOAD_FROM_EMA_ON_START = True

    # Begin domain adaptation settings
    _C.DA = CN()

    # Source-target alignment
    _C.DA.ALIGN = CN()

    # RPN Domain Adaptation
    _C.DA.ALIGN.RPN_DA_ENABLED = False
    _C.DA.ALIGN.RPN_DA_WEIGHT = 0.01
    _C.DA.ALIGN.RPN_DA_INPUT_DIM = 256
    _C.DA.ALIGN.RPN_DA_HIDDEN_DIMS = [256,]

    # Style Transfer
    _C.DA.ALIGN.STYLE_ENABLED = False
    _C.DA.ALIGN.STYLE_ALPHA = 0.3
    _C.DA.ALIGN.STYLE_LEVELS = ["p2", "p3"]
    _C.DA.ALIGN.STYLE_STAT_WEIGHT = 0.01
    _C.DA.ALIGN.STYLE_STRENGTH = 0.5

    # Self-distillation
    _C.DA.DISTILL = CN()
    _C.DA.DISTILL.DISTILLER_NAME = "DA_F2FDistiller"
    # 'Pseudo label' approaches
    _C.DA.DISTILL.HARD_ROIH_CLS_ENABLED = False
    _C.DA.DISTILL.HARD_ROIH_REG_ENABLED = False
    _C.DA.DISTILL.HARD_OBJ_ENABLED = False
    _C.DA.DISTILL.HARD_RPN_REG_ENABLED = False
    # 'Distillation' approaches
    _C.DA.DISTILL.ROIH_CLS_ENABLED = False
    _C.DA.DISTILL.ROIH_REG_ENABLED = False
    _C.DA.DISTILL.OBJ_ENABLED = False
    _C.DA.DISTILL.RPN_REG_ENABLED = False
    _C.DA.DISTILL.CLS_TMP = 1.0
    _C.DA.DISTILL.OBJ_TMP = 1.0
    _C.DA.CLS_LOSS_TYPE = "CE"

    _C.DA.TEACHER = CN()
    _C.DA.TEACHER.ENABLED = False
    _C.DA.TEACHER.THRESHOLD = 0.8

    # Solver
    _C.SOLVER.IMS_PER_GPU = 2
    _C.SOLVER.BACKWARD_AT_END = True
    _C.SOLVER.CLIP_GRADIENTS.ENABLED = True
    _C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    _C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 10.0

    _C.SOLVER.OPTIMIZER = "SGD"