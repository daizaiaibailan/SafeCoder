import os
import sys
import torch

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sven.model import load_model
from sven.utils import set_logging, set_devices


class Args:
    pass


args = Args()

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
os.makedirs(os.path.join(root_dir, 'trained', 'demo_logs'), exist_ok=True)
set_logging(args, os.path.join(root_dir, 'trained', 'demo_logs', 'demo.log'))
set_devices(args)

print('Loading prefix model...')
tokenizer, model, device = load_model(
    'prefix',
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
                 'trained', '350m-prefix-new', 'checkpoint-last'),
    False, args
)
model.eval()
print('Done.\n')

prompt = 'def query_user(name):\n    sql = "SELECT * FROM users WHERE name = \'" + name + "\'"\n    cursor.execute(sql)'

print('=== INPUT ===')
print(prompt)
print()

inputs = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

print('=== control_id=0 (sec - secure generation) ===')
out = model.generate(inputs, do_sample=True, max_new_tokens=80, temperature=0.4,
                     control_id=0, pad_token_id=tokenizer.pad_token_id)
gen = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
print(prompt + gen)
print()

print('=== control_id=1 (vul - vulnerable generation) ===')
out = model.generate(inputs, do_sample=True, max_new_tokens=80, temperature=0.4,
                     control_id=1, pad_token_id=tokenizer.pad_token_id)
gen = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
print(prompt + gen)
