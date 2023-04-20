from . import inject_tfkeras_modules, init_tfkeras_custom_objects
from . import model
# !git clone https://github.com/google/automl.git
import sys
sys.path.insert(0,'/kaggle/working/automl/efficientnetv2')
import effnetv2_model
from .preprocessing import center_crop_and_resize
net = effnetv2_model.get_model('efficientnetv2-m', include_top=False)
EfficientNetB0 = inject_tfkeras_modules(model.EfficientNetB0)
EfficientNetB1 = inject_tfkeras_modules(model.EfficientNetB1)
EfficientNetB2 = inject_tfkeras_modules(model.EfficientNetB2)
EfficientNetB3 = inject_tfkeras_modules(model.EfficientNetB3)
EfficientNetB4 = inject_tfkeras_modules(model.EfficientNetB4)
EfficientNetB5 = inject_tfkeras_modules(model.EfficientNetB5)
EfficientNetB6 = inject_tfkeras_modules(model.EfficientNetB6)
EfficientNetB7 = inject_tfkeras_modules(model.EfficientNetB7)
EfficientNetL2 = inject_tfkeras_modules(model.EfficientNetL2)
EfficientNetL3 = inject_tfkeras_modules(model.EfficientNetL3)

preprocess_input = inject_tfkeras_modules(model.preprocess_input)

init_tfkeras_custom_objects()
