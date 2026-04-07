import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_densepose_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_densepose_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("000000153299.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="giraffe")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]