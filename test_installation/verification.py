from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv

# Specify the configuration file for the model.
config_file = '../configs/rtmdet_tiny_8xb32-300e_coco.py'

# Specify the pre-trained model checkpoint file.
checkpoint_file = '../configs/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

# Path to the input image on which inference will be performed.
image_path = '../demo/demo.jpg'

# Initialize the detection model with the specified configuration and checkpoint.
# The device can be set to 'cuda:0' for GPU usage or 'cpu' if a GPU is not available.
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Perform inference on the input image to detect objects.
# The result contains the detection information (e.g., bounding boxes, confidence scores).
result = inference_detector(model, image_path)

# Create a visualizer instance to render the detection results on the input image.
visualizer = DetLocalVisualizer()

# Set the metadata for the dataset, including the class names, for proper labeling.
visualizer.dataset_meta = {'classes': model.dataset_meta['classes']}

# Read the input image in RGB format.
# The 'channel_order' parameter ensures the image is loaded in the correct format.
image = mmcv.imread(image_path, channel_order='rgb')

# Add the detection results to the visualizer and save the annotated image.
# - name: A label for the dataset being visualized.
# - image: The original image to be annotated.
# - data_sample: The detection results from the inference step.
# - draw_gt: Whether to draw ground truth annotations (False since this is for predictions).
# - show: Whether to display the annotated image (False to save instead).
# - pred_score_thr: Confidence threshold to filter low-confidence predictions.
# - out_file: File path to save the annotated output image.
visualizer.add_datasample(
    name='result',
    image=image,
    data_sample=result,
    draw_gt=False,
    show=False,
    pred_score_thr=0.5,
    out_file='../demo/demo_result.jpg'
)

# Print a confirmation message indicating where the results have been saved.
print("Detection results saved to 'demo/demo_result.jpg'")
