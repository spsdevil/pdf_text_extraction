print("\n\n\t|---------------------:WELCOME:---------------------|")
print("\t|----------------------HANG ON----------------------|")
# Imports
import warnings
warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from utils import backbone
import complete3

print("\t|----------------Alright, I'm Ready-----------------|\n\n")
# Set Model---------------------------------------------------------------------------------------------------------------------
detection_graph, category_index = backbone.set_model('custom_frozen_inference_graph', 'labelmap.pbtxt')


complete3.pdf_to_text(detection_graph, category_index)