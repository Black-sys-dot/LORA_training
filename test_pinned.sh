pip install diffusers==0.29.2 transformers==4.41.2 accelerate==0.31.0 peft==0.11.1 datasets==2.20.0 "pyarrow<17.0.0" "numpy<2.0.0"
python3 -c "
import diffusers
import transformers
import accelerate
import datasets
import peft
from peft import LoraConfig
print('ALL IMPORTS WORKED SUCCESSFULLY')
"
