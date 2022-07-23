from torchvision.models import vgg16

from model.utils.creators import AnchorTargetCreator, ProposalTargetCreator

from model.fasetr_rcnn import RegionProposalNetwork, FasterRcnn, ProposalCreator
from model.faster_rcnn_vgg16 import decom_vgg16, FasterRcnnVgg16
from model.faster_rcnn_trainer import FasterRcnnTrainer



def main():
    extractor = vgg16(pretrained=True).features[:-1]

    test_sub_module = "faster_rcnn"

    from data.data_interface import DInterface

    dataset = DInterface(data_dir=r'test_data/')
    dataset.setup(stage="fit")
    imgs, gt_bboxes, gt_labels, diffs, scale = dataset.train_set[0]
    img_size = imgs.shape[1:]
    imgs = imgs.unsqueeze(0)  # Simulate one batch with batch size 1
    
    val_imgs, val_gt_bboxes, val_gt_labels, val_diffs, val_scale = dataset.val_set[0]
    val_img_size = imgs.shape[1:]
    val_imgs = imgs.unsqueeze(0)  # Simulate one batch with batch size 1
    # H, W of the original image

    if test_sub_module == "extractor":
        # For 000005.jpg
        print(f"Scale is {scale}")  # 1.6
        print(imgs.shape)  # (3, 600, 800)
        output = extractor(imgs)
        print(output.shape)  # (512, 37, 50)
    elif test_sub_module == "rpn":  # Include Proposal Creator
        rpn = RegionProposalNetwork()
        features = extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_idx, shifted_anchors = rpn(features, img_size, scale)
        print(rpn.anchor_base)
    elif test_sub_module == "anchor_target_creator":
        rpn = RegionProposalNetwork()
        features = extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_idx, shifted_anchors = rpn(features, img_size, scale)
        atc = AnchorTargetCreator()
        gt_locs, created_labels = atc(gt_bboxes, shifted_anchors, img_size)
        print(gt_locs, created_labels)
    elif test_sub_module == "proposal_target_creator":
        rpn = RegionProposalNetwork()
        features = extractor(imgs)
        rpn_locs, rpn_scores, RoIs, RoI_idx, shifted_anchors = rpn(features, img_size, scale)
        ptc = ProposalTargetCreator()
        RoI_samples, gt_RoI_locs, gt_RoI_labels = ptc(RoIs, gt_bboxes, gt_labels)
        print(RoI_samples, gt_RoI_locs, gt_RoI_labels)
    elif test_sub_module == "decomposed_vgg16":
        extractor, classifier = decom_vgg16(pretrained=True, use_drop=True)
        print(extractor, classifier)
        for name, param in extractor.named_parameters():
            if param.requires_grad:
                print(name)
    elif test_sub_module == "faster_rcnn_vgg16":
        model = FasterRcnnVgg16()
        (
            rpn_locs, rpn_scores,
            gt_rpn_locs, gt_rpn_labels,
            roi_cls_locs, roi_scores,
            gt_roi_locs, gt_roi_labels,
            rois
        ) = model(imgs, gt_bboxes, gt_labels, scale=scale)
        # print(roi_scores.shape)
        # print(roi_cls_locs)
        # print(model.extractor)

    # elif test_sub_module == "faster_rcnn_trainer":
    #     trainer = FasterRcnnTrainer()
    #     print(trainer.__dict__)
    elif test_sub_module == "faster_rcnn":
        model = FasterRcnnVgg16()
        
        # Val
        model.eval()
        model.
        
        

    elif test_sub_module == "proposal_creator":
        # Use a model to generate locs, scores, anchors
        parent_model = None
        proposal_creator = ProposalCreator(parent_model=parent_model)


if __name__ == "__main__":
    main()