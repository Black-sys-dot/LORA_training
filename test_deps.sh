pip install diffusers transformers accelerate datasets peft "pyarrow<17.0.0" "numpy<2.0.0"
python3 -c "
import diffusers
import transformers
import accelerate
import datasets
import peft
print('ALL IMPORTS WORKED SUCCESSFULLY')
"
