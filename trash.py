# ./data/pascal_voc.py
# USE OPENCV TO TEST THE DATA AND DRAW BOUNDING BOXES
# def draw_bboxes_labels(img, bboxes, labels, label_name):
#     color = (255, 0, 0)
#     thickness = 1
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 0.5
#     for i in range(len(bboxes)):
#         bbox = bboxes[i]
#         label = labels[i]
#         cv2.rectangle(
#             img,
#             (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])),
#             color,
#             thickness
#         )
#         cv2.putText(
#             img,
#             VOC_BBOX_LABEL_NAMES[label],
#             (int(bbox[1]) + 5, int(bbox[2]) - 5),
#             font,
#             fontScale,
#             color,
#             thickness
#         )
#
#     cv2.imshow("Test", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# ./data/data_utils.py
# READ IMAGE
# def imread(img_path, dtype=torch.float32, color=True):
#     """
#
#     Args:
#         img_path: The file path of the image
#         dtype: The type of image data, default is torch.float32
#         color: When input a color image, it's "True"
#
#     Returns:
#
#     """
#     img = Image.open(img_path)
#     if not color:
#         img =img
#
#     return img


# def draw_bboxes_labels(img, bboxes, labels, label_name):
#     img_copy = img.detach()
#     img_copy = img_copy.permute((1, 2, 0))
#     img_copy = torch.clamp(img_copy*255, 0, 255).type(torch.uint8).numpy()
#
#     print(img_copy.shape)
#     print(img_copy.dtype)
#     print(type(img_copy))
#     color = (255, 0, 0)
#     thickness = 1
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 0.5
#     for i in range(len(bboxes)):
#         bbox = bboxes[i]
#         label = labels[i]
#         cv2.rectangle(
#             img_copy,
#             (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])),
#             color,
#             thickness
#         )
#         cv2.putText(
#             img_copy,
#             label_name[label],
#             (int(bbox[1]) + 5, int(bbox[2]) - 5),
#             font,
#             fontScale,
#             color,
#             thickness
#         )
#     cv2.imshow("Test", img_copy)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# ../data/pascal_voc.py
# Draw bboxes for the image
# def draw_bboxes(draw, bboxes):
#     for i in range(len(bboxes)):
#         bbox = bboxes[i]
#         draw.rectangle(
#             [(int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2]))],
#             outline="red"
        )