import sys
import os
import subprocess
import numpy as np
import pandas as pd

"""
    脚本使用：
        环境准备: 1. 安装好 opt 工具  2. python3
        执行： nohup python3 paddle_model_operator_statistics_tool.py &
        参数说明：
            1. need_download_models #是否需要重新下载模型，此过程很慢，如果已经下载过将该值设置成False
            2. opt_dir #opt工具所在的路径
            5. output_dir #结果保存目录
        脚本执行结果: 会在 output_dir 目录下生成 op_support_model.csv 文件, 里面统计了 模型-算子 的支持情况
        新增模型: 只需要在相应类下面加入新的模型链接
        待统计模型修改: 参考main函数的代码，修改相应统计的模型list即可
"""

# 脚本运行选项设置
need_download_models=True  #是否需要重新下载模型，此过程很慢，如果已经下载过将该值设置成False
opt_dir = "/Users/chensiyu08/Library/Python/2.7/bin/paddle_lite_opt"  #opt工具目录
output_dir = "./output"


class ModelsCollection(object):

    def GetAllModelsDic(self):
        return vars(self).keys()


class PaddleClasModels(ModelsCollection):
    """
        ************** PaddleClas Models **************
    """
    def __init__(self):
        # super().__init__()
        self.SE_RESNET50 = {"name" : ["RESNET50"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/SENet/SE_ResNet50_vd.tar.gz"]}
        self.HRNet_W18_C = {"name" : ["HRNet_W18_C"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/HRNet/HRNet_W18_C.tar.gz"]}
        self.ViT_base_patch16_224 = {"name" : ["ViT_base_patch16_224"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/VisionTransformer/ViT_base_patch16_224.tar.gz"]}
        self.DarkNet53 = {"name" : ["DarkNet53"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DarkNet/DarkNet53.tar.gz"]}
        self.SwinTransformer_base_patch4_window12_384 = {"name" : ["SwinTransformer_base_patch4_window12_384"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/SwinTransformer/SwinTransformer_base_patch4_window12_384.tar.gz"]}
        self.DPN68 = {"name" : ["DPN68"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DPN/DPN68.tar.gz"]}
        self.DeiT_base_patch16_224 = {"name" : ["DeiT_base_patch16_224"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/DeiT/DeiT_base_patch16_224.tar.gz"]}
        self.GhostNet_x1_0 = {"name" : ["GhostNet_x1_0"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/GhostNet/GhostNet_x1_0.tar.gz"]}
        self.Res2Net50_26w_4s = {"name" : ["Res2Net50_26w_4s"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/Res2Net/Res2Net50_26w_4s.tar.gz"]}
        self.PPLCNet_x0_25 = {"name" : ["PPLCNet_x0_25"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/v2.3/PPLCNet/PPLCNet_x0_25.tar.gz"]}
        self.AlexNet = {"name" : ["AlexNet"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/AlexNet.tgz"]}
        self.DenseNet121 = {"name" : ["DenseNet121"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/DenseNet121.tgz"]}
        self.GoogLeNet = {"name" : ["GoogLeNet"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/GoogLeNet.tgz"]}
        self.Inception_v3 = {"name" : ["Inception_v3"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV3.tgz"]}
        self.Inception_v4 = {"name" : ["Inception_v4 "], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/InceptionV4.tgz"]}
        self.MnasNet = {"name" : ["MnasNet"], "url" : [""]}
        self.MobileNet_v1 = {"name" : ["MobileNet_v1"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV1.tgz"]}
        self.MobileNet_v2 = {"name" : ["MobileNet_v2 "], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV2.tgz"]}
        self.MobileNetV3_large_x1_0 = {"name" : ["MobileNetV3_large_x1_0"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/MobileNetV3_large_x1_0.tgz"]}
        self.ResNet_101 = {"name" : ["ResNet_101"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet101.tgz"]}
        self.ResNet_18 = {"name" : ["ResNet_18"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet18.tgz"]}
        self.ResNet_50  = {"name" : ["ResNet_50 "], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNet50.tgz"]}
        self.ResNeXt50 = {"name" : ["ResNeXt50"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ResNeXt50_32x4d.tgz"]}
        self.ShuffleNetv2 = {"name" : ["ShuffleNetv2"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/ShuffleNetV2_x1_0.tgz"]}
        self.SqueezeNet_v1 = {"name" : ["SqueezeNet_v1"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/SqueezeNet1_0.tgz"]}
        self.VGG16 = {"name" : ["VGG16"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG16.tgz"]}
        self.VGG19 = {"name" : ["VGG19"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/VGG19.tgz"]}
        self.EfficientNet_b0 = {"name" : ["EfficientNet_b0"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleClas/EfficientNetB0.tgz"]}

    def GetUseOptTransModelsList(self):
        use_opt_trans_models_list = [self.SE_RESNET50, self.HRNet_W18_C, self.ViT_base_patch16_224, self.DarkNet53,
            self.SwinTransformer_base_patch4_window12_384, self.DPN68, self.DeiT_base_patch16_224, self.GhostNet_x1_0,
            self.Res2Net50_26w_4s, self.PPLCNet_x0_25]
        return use_opt_trans_models_list

class PaddleSegModels(ModelsCollection):
    """
        ************** PaddleSeg Models **************
    """
    def __init__(self):
        # super().__init__()
        self.deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax = {"name" : ["deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax.tar.gz"]}
        self.pphumanseg_lite_generic_192x192_with_softmax = {"name" : ["pphumanseg_lite_generic_192x192_with_softmax"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/pphumanseg_lite_generic_192x192_with_softmax.tar.gz"]}
        self.OCRNet = {"name" : ["OCRNet"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/ocrnet.tar.gz"]}
        self.BiSeNetv2 = {"name" : ["BiSeNetv2"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/bisenet.tar.gz"]}
        self.SegFormer = {"name" : ["SegFormer"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/segformer.tar.gz"]}
        self.STDC = {"name" : ["STDC"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/stdcseg.tar.gz"]}
        self.U_Net = {"name" : ["U_Net"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/unet_cityscapes_1024x512_160k.tar.gz"]}
        self.DeepLabV3 = {"name" : ["DeepLabV3"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleSeg/v2.3/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.tar.gz"]}

    def GetUseOptTransModelsList(self):
        use_opt_trans_models_list = [self.deeplabv3p_resnet50_os8_humanseg_512x512_100k_with_softmax, 
            self.pphumanseg_lite_generic_192x192_with_softmax,
            self.OCRNet, self.BiSeNetv2, self.SegFormer, self.STDC]
        return use_opt_trans_models_list

class PaddleOCRModels(ModelsCollection):
    """
        ************** PaddleOCR Models **************
    """
    def __init__(self):
        # super().__init__()
        self.ch_ppocr_server_v2_0_rec = {"name" : ["ch_ppocr_server_v2_0_rec"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_rec_infer.tar.gz"]}
        self.ch_ppocr_server_v2_0_det = {"name" : ["ch_ppocr_server_v2_0_det"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_server_v2.0_det_infer.tar.gz"]}
        self.ch_PP_OCRv2_det = {"name" : ["ch_PP_OCRv2_det"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_det_infer.tar.gz"]}
        self.ch_PP_OCRv2_rec = {"name" : ["ch_PP_OCRv2_rec"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_PP-OCRv2_rec_infer.tar.gz"]}
        self.crnn_ctc = {"name" : ["crnn_ctc"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/rec_crnn_mv3_ctc.tar.gz"]}
        self.e2e = {"name" : ["e2e"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/e2e_server_pgnetA.tar.gz"]}
        self.OCR_DB = {"name" : ["OCR_DB "], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_mobile_v2_0_det_v2_0.tar.gz"]}
        self.OCR_Clas = {"name" : ["OCR_Clas"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleOCR/v2.3/ch_ppocr_mobile_v2_0_rec_v2_0.tar.gz"]}

        ### Slim Model
        self.ch_PP_OCRv2_det_slim = {"name" : ["ch_PP_OCRv2_det_slim"], "url" : ["https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar"]}
        self.ch_PP_OCRv2_rec_slim = {"name" : ["ch_PP_OCRv2_rec_slim"], "url" : ["https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar"]}
        self.ch_ppocr_mobile_slim_v2_0_det = {"name" : ["ch_ppocr_mobile_slim_v2_0_det"], "url" : ["https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar"]}
        self.ch_ppocr_mobile_slim_v2_0_rec = {"name" : ["ch_ppocr_mobile_slim_v2_0_rec"], "url" : ["https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar"]}

    def GetUseOptTransModelsList(self):
        use_opt_trans_models_list = [
            self.ch_ppocr_server_v2_0_rec,
            self.ch_ppocr_server_v2_0_det,
            self.ch_PP_OCRv2_det,
            self.ch_PP_OCRv2_rec,
            self.crnn_ctc,
            self.e2e
        ]
        return use_opt_trans_models_list

class PaddleDetectionModels(ModelsCollection):
    """
        ************** PaddleDetection Models **************
    """
    def __init__(self):
        # super().__init__()
        self.ssdlite_mobilenet_v3_large = {"name" : ["ssdlite_mobilenet_v3_large"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_large.tar.gz"]}
        self.ssdlite_mobilenet_v3_small = {"name" : ["ssdlite_mobilenet_v3_small"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/static/ssdlite_mobilenet_v3_small.tar.gz"]}
        self.ppyolo_tiny_650e_coco = {"name" : ["ppyolo_tiny_650e_coco"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_tiny_650e_coco.tar.gz"]}
        self.ppyolo_mbv3_large_coco = {"name" : ["ppyolo_mbv3_large_coco"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_mbv3_large_coco.tar.gz"]}
        self.ppyolo_r50vd_dcn_1x_coco = {"name" : ["ppyolo_r50vd_dcn_1x_coco"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolo_r50vd_dcn_1x_coco.tar.gz"]}
        self.ppyolov2_r50vd_dcn_365e_coco = {"name" : ["ppyolo_tiny_650e_coco"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/ppyolov2_r50vd_dcn_365e_coco.tar.gz"]}
        self.picodet_m_416_coco = {"name" : ["picodet_m_416_coco"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/picodet_m_416_coco.tar.gz"]}
        self.tinypose_128x96 = {"name" : ["tinypose_128x96"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/tinypose_128x96.tar.gz"]}
        self.FairMOT = {"name" : ["FairMOT"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/fairmot_dla34_30e_1088x608.tar.gz"]}
        self.JDE = {"name" : ["JDE"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/jde_darknet53_30e_1088x608.tar.gz"]}
        self.Cascade_Faster_RCNN = {"name" : ["Cascade_Faster_RCNN"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/cascade_rcnn_r50_fpn_1x_coco.tar.gz"]}
        self.Cascade_Mask_RCNN_FPN = {"name" : ["Cascade_Mask_RCNN_FPN"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/cascade_mask_rcnn_r50_fpn_gn_2x_coco.tar.gz"]}
        self.Cascade_Mask_RCNN = {"name" : ["Cascade_Mask_RCNN"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/cascade_mask_rcnn_r50_fpn_1x_coco.tar.gz"]}
        self.Faster_RCNN = {"name" : ["Faster_RCNN"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/faster_rcnn_r50_1x_coco.tar.gz"]}
        self.Faster_RCNN_FPN = {"name" : ["Faster_RCNN_FPN"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/faster_rcnn_r50_fpn_1x_coco.tar.gz"]}
        self.Mask_RCNN = {"name" : ["Mask_RCNN"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/v2.3/mask_rcnn_r50_1x_coco.tar.gz"]}
        self.BlazeFace = {"name" : ["BlazeFace"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/blazeface_1000e.tgz"]}
        self.Faceboxes = {"name" : ["Faceboxes"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/faceboxes.tgz"]}
        self.HigherHRNet = {"name" : ["HigherHRNet"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/higherhrnet_hrnet_w32_640.tgz"]}
        self.HRNet = {"name" : ["HRNet"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/hrnet_w32_384x288.tgz"]}
        self.PP_YOLO = {"name" : ["PP_YOLO"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ppyolo_r50vd_dcn_1x_coco.tgz"]}
        self.SSD_MobileNetV1 = {"name" : ["SSD_MobileNetV1"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_mobilenet_v1_300_120e_voc.tgz"]}
        self.SSD_VGG16 = {"name" : ["SSD_VGG16"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/ssd_vgg16_300_240e_voc.tgz"]}
        self.YOLOv3_DarkNet53 = {"name" : ["YOLOv3_DarkNet53"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_darknet53_270e_coco.tgz"]}
        self.YOLOv3_MobileNetV1 = {"name" : ["YOLOv3_MobileNetV1"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v1_270e_coco.tgz"]}
        self.YOLOv3_MobileNetV3 = {"name" : ["YOLOv3_MobileNetV3"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_mobilenet_v3_large_270e_coco.tgz"]}
        self.YOLOv3_ResNet50_vd = {"name" : ["YOLOv3_ResNet50_vd"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov3_r50vd_dcn_270e_coco.tgz"]}
        self.YOLOv4 = {"name" : ["YOLOv4 "], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleDetection/yolov4_cspdarknet.tgz"]}

    def GetUseOptTransModelsList(self):
        use_opt_trans_models_list = [self.ssdlite_mobilenet_v3_large, self.ssdlite_mobilenet_v3_small, self.ppyolo_tiny_650e_coco,
            self.ppyolo_mbv3_large_coco, self.ppyolo_r50vd_dcn_1x_coco, self.ppyolov2_r50vd_dcn_365e_coco, self.picodet_m_416_coco,
            self.tinypose_128x96, self.Cascade_Faster_RCNN, self.Cascade_Mask_RCNN_FPN, self.Cascade_Mask_RCNN, self.Faster_RCNN,
            self.Faster_RCNN_FPN, self.Mask_RCNN]
        return use_opt_trans_models_list
    
class PaddleVideoModels(ModelsCollection):
    """
        ************** PaddleVideo Models **************
    """
    def __init__(self):
        # super().__init__()
        self.ppTSM = {"name" : ["ppTSM"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSM.tar.gz"]}
        self.ppTSN = {"name" : ["ppTSN"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleVideo/v2.2.0/ppTSN.tar.gz"]}

    def GetUseOptTransModelsList(self):
        use_opt_trans_models_list = [self.ppTSM, self.ppTSN]
        return use_opt_trans_models_list

class PaddleSpeechModels(ModelsCollection):
    """
        ************** PaddleSpeech Models **************
    """
    def __init__(self):
        # super().__init__()
        self.SpeedySpeech = {"name" : ["SpeedySpeech"], "url" : ["https://paddlespeech.bj.bcebos.com/Parakeet/speedyspeech_nosil_baker_static_0.5.zip"]}
        self.FastSpeech2 = {"name" : ["FastSpeech2"], "url" : ["https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech2_nosil_baker_static_0.4.zip"]}
        self.Parallel_Wave_GAN = {"name" : ["Parallel_Wave_GAN"], "url" : ["https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_static_0.4.zip"]}
        self.Multi_band_MelGAN = {"name" : ["Multi_band_MelGAN"], "url" : ["https://paddlespeech.bj.bcebos.com/Parakeet/mb_melgan_baker_static_0.5.zip"]}

    def GetUseOptTransModelsList(self):
        use_opt_trans_models_list = [
            self.SpeedySpeech,
            # self.FastSpeech2,
            # self.Parallel_Wave_GAN,
            # self.Multi_band_MelGAN
        ]
        return use_opt_trans_models_list

class PaddleRecModels(ModelsCollection):
    """
        ************** PaddleRec Models **************
    """
    def __init__(self):
        # super().__init__()
        self.DeepFM = {"name" : ["DeepFM"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/deepfm.tar.gz"]}
        self.NAML = {"name" : ["NAML"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/naml.tar.gz"]}
        self.NCF = {"name" : ["NCF"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/ncf.tar.gz"]}
        self.WideDeep = {"name" : ["WideDeep"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleRec/v2.1.0/wide_deep.tar.gz"]}

    def GetUseOptTransModelsList(self):
        use_opt_trans_models_list = [
            self.DeepFM,
            self.NAML,
            self.NCF,
            self.WideDeep
        ]
        return use_opt_trans_models_list

class PaddleGanModels(ModelsCollection):
    """
        ************** PaddleGan Models **************
    """
    def __init__(self):
        # super().__init__()
        self.ESRGAN = {"name" : ["BlazeFace"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleGAN/esrgan_psnr_x4_div2k.tar.gz"]}

    def GetUseOptTransModelsList(self):
        use_opt_trans_models_list = [self.ESRGAN]
        return use_opt_trans_models_list

class PaddleNLPModels(ModelsCollection):
    """
        ************** PaddleNLP Models **************
    """
    def __init__(self):
        # super().__init__()
        self.BERT = {"name" : ["BERT"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/bert_base_uncased.tgz"]}
        self.ERNIE = {"name" : ["ERNIE "], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_1.0.tgz"]}
        self.ERNIE_TINY = {"name" : ["ERNIE_TINY"], "url" : ["https://paddlelite-demo.bj.bcebos.com/NNAdapter/models/PaddleNLP/ernie_tiny.tar.gz"]}

    def GetUseOptTransModelsList(self):
        use_opt_trans_models_list = [self.BERT, self.ERNIE, self.ERNIE_TINY]
        return use_opt_trans_models_list


class ModelOptTools(object):
    def __init__(self, model_list, need_download_models, opt_dir, output_dir = "./"):
        self.model_list = model_list
        self.need_download_models = need_download_models
        self.opt_dir = opt_dir
        self.output_dir = output_dir
        
        self.save_models_dir = os.path.join(output_dir, "download_models")
        self.save_model_txt_dir = os.path.join(output_dir, "generate_text")
        self.support_model_list = []
        self.not_support_model_list = []
        self.model_ops_dic = {}
        
    def Prepare(self):
        if not os.path.exists(self.output_dir):
            subprocess.Popen("mkdir -p " + self.output_dir, stdout=subprocess.PIPE, shell=True)
        if not os.path.exists(self.save_models_dir):
            subprocess.Popen("mkdir -p " + self.save_models_dir, stdout=subprocess.PIPE, shell=True)
        if not os.path.exists(self.save_model_txt_dir):
            subprocess.Popen("mkdir -p " + self.save_model_txt_dir, stdout=subprocess.PIPE, shell=True)

    # 将所有models.csv文件中的op-name解析出来，并标记出使用该算子的模型
    def SelectOpsAndModels(self):
        for root, model_op_info_dir, model_op_info_files in os.walk(self.save_model_txt_dir):
            for model_op_info_file in model_op_info_files:
                csv_op_info_file_path = os.path.join(self.save_model_txt_dir, model_op_info_file)
                with open(csv_op_info_file_path) as f:
                    # 先判断文件是否可用
                    lines = f.readlines()
                    if "Paddle-Lite supports this model!" in lines[-1]:
                        self.support_model_list.append("".join(model_op_info_file.split(".")[0:-1]))
                    else:
                        self.not_support_model_list.append("".join(model_op_info_file.split(".")[0:-1]))
                        continue

                    for line in lines:
                        if ("OPs in the input model include" or "Paddle-Lite supports this model!") in line:
                            continue
                        line = line.strip().split(" ")
                        op_name = line[0]
                        if op_name.startswith("__"):                          
                            op_name = op_name[2:]
                        if op_name not in self.model_ops_dic:
                            self.model_ops_dic[op_name] = []
                            self.model_ops_dic[op_name].append("".join(model_op_info_file.split(".")[0:-1]))
                        else:
                            if model_op_info_file not in self.model_ops_dic[op_name]:
                                self.model_ops_dic[op_name].append("".join(model_op_info_file.split(".")[0:-1]))
        

    def GetModelOpsDic(self):
        return self.model_ops_dic        

    def excuteCommand(self, cmd):
        print("execute:", cmd)
        ex = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out, err = ex.communicate()
        status = ex.wait()
        return out.decode()
    
    def OptGenCsv(self):
        self.Prepare()
        self.UseOptGenModelOps()
        self.SelectOpsAndModels()
        model_ops_dataframe = pd.DataFrame({"op":self.model_ops_dic.keys(), "model_name":self.model_ops_dic.values()})
        output_csv_file = os.path.join(output_dir, "op_support_model.csv")
        model_ops_dataframe.to_csv(output_csv_file, index = False)
        print("Save op csv file success!")
        print("These models is supported:", " ".join(self.support_model_list))
        print("These models is not supported:", " ".join(self.not_support_model_list))

    def DownloadModels(self, models_dataframe):
        names = models_dataframe["name"]
        urls = models_dataframe["url"]
        compressed_model_path = os.path.join(self.save_models_dir, "compressed_model")
        if not os.path.exists(compressed_model_path):
            subprocess.Popen("mkdir -p " + compressed_model_path, stdout=subprocess.PIPE, shell=True)
        uncompressed_model_path = os.path.join(self.save_models_dir, "uncompressed_model")
        if not os.path.exists(uncompressed_model_path):
            subprocess.Popen("mkdir -p " + uncompressed_model_path, stdout=subprocess.PIPE, shell=True)
       
        for url in urls:
            if url != "":
                model_tar_name = url.strip("").split("/")[-1]
                model_name = "".join(model_tar_name.split(".")[0:-1])
                model_dir_path = os.path.join(uncompressed_model_path, model_name)
                if os.path.exists(model_dir_path):
                    continue
                model_tar_file_path = os.path.join(compressed_model_path, model_tar_name)
                if not os.path.exists(model_tar_file_path):
                    self.excuteCommand('wget -P ' + compressed_model_path + " " + url)
                if model_tar_name.endswith(".tar.gz") or model_tar_name.endswith(".tgz"):
                    self.excuteCommand("tar -zxvf " + model_tar_file_path + " -C " + uncompressed_model_path)
                elif model_tar_name.endswith(".tar"):
                    self.excuteCommand("tar -xvf " + model_tar_file_path + " -C " + uncompressed_model_path)
                elif model_tar_name.endswith(".zip"):
                    self.excuteCommand("unzip -o " + model_tar_file_path + " -d " + uncompressed_model_path)
                else:
                    pass

    
    def UseOptGenModelOps(self):
        """
            利用 opt 工具统计所有模型的算子
        """
        # 1. Create dataframe
        model_dataframe = pd.DataFrame(columns=["name", "url"])
        for model_name in self.model_list:
            model_dataframe = model_dataframe.append(pd.DataFrame(model_name), ignore_index=True)

        # 2. Download and uncompress models
        if self.need_download_models:
            self.DownloadModels(model_dataframe)  

        # 3. Opt Trans
        for fpath, dirname, fnames in os.walk(self.save_models_dir):
            model_name = fpath.split("/")[-1]
            pd_model_name = ""
            pd_params_name = ""
            for name in fnames:
                if name.endswith("pdmodel") or name == "__model__":
                    pd_model_name = name
                if name.endswith("pdiparams") or name == "__params__":
                    pd_params_name = name
                
            if pd_model_name == "" or pd_params_name == "":
                print("%s dir doesn\'t has pdmodel or pdiparams file, check it!" % model_name)
                continue
            pd_model_file_path = os.path.join(fpath, pd_model_name)
            pd_params_file_path = os.path.join(fpath, pd_params_name)
            print(pd_model_file_path, pd_params_file_path)
            opt_cmd = opt_dir + " --model_file=" +  pd_model_file_path + \
                                " --param_file=" +  pd_params_file_path + \
                                " --print_model_ops=true > " + os.path.join(self.save_model_txt_dir, model_name) + ".txt"
            self.excuteCommand(opt_cmd)
            print("\n")
    


if __name__ == '__main__':
    # 1. 根据各个 Repo 下面收集的模型列表制作需要用 opt 工具转换的模型列表 opt_trans_model_list
    print("*** Step 1: Create opt trans model list ***")
    paddle_clas_models = PaddleClasModels()
    paddle_detection_models = PaddleDetectionModels()
    paddle_seg_models = PaddleSegModels()
    paddle_speech_models = PaddleSpeechModels()
    paddle_rec_models = PaddleRecModels()
    paddle_video_models = PaddleVideoModels()
    paddle_ocr_models = PaddleOCRModels()
    paddle_gan_models = PaddleGanModels()
    paddle_nlp_models = PaddleNLPModels()

    opt_trans_model_list = [] 
    opt_trans_model_list.extend(paddle_clas_models.GetUseOptTransModelsList())
    opt_trans_model_list.extend(paddle_detection_models.GetUseOptTransModelsList())
    opt_trans_model_list.extend(paddle_seg_models.GetUseOptTransModelsList())
    opt_trans_model_list.extend(paddle_speech_models.GetUseOptTransModelsList())
    opt_trans_model_list.extend(paddle_rec_models.GetUseOptTransModelsList())
    opt_trans_model_list.extend(paddle_video_models.GetUseOptTransModelsList())
    opt_trans_model_list.extend(paddle_ocr_models.GetUseOptTransModelsList())
    opt_trans_model_list.extend(paddle_gan_models.GetUseOptTransModelsList())
    opt_trans_model_list.extend(paddle_nlp_models.GetUseOptTransModelsList())
    print("There are {} models in the opt_trans_model_list!".format(len(opt_trans_model_list)))
    
    # 2. 通过 ModelOptTools 进行转换, 生成 op_support_model.csv 文件在目标目录下, 默认当前目录
    print("*** Step 2: Download and uncompress models, use opt tool statistic! ***")
    model_opt_tools = ModelOptTools(
        model_list = opt_trans_model_list,
        need_download_models = need_download_models,
        opt_dir = opt_dir,
        output_dir = output_dir
        )
    model_opt_tools.OptGenCsv()
