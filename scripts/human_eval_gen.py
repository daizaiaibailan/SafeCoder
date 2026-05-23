import os
import argparse
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from sven.model import load_model
from sven.human_eval.problem_yaml import Problem
from sven.utils import set_logging, set_devices, set_seed


def main():
    parser = argparse.ArgumentParser(description='SVEN HumanEval generation')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['lm', 'prefix', 'text-prompt'])
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--control', type=str, default=None,
                        choices=['sec', 'vul'],
                        help='Control label for prefix model. If not set, uses lm mode without prefix.')
    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--num_gen', type=int, default=100)
    parser.add_argument('--max_gen_len', type=int, default=512)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    args.data_dir = os.path.join(root_dir, 'data_eval', 'human_eval')
    args.output_dir = os.path.join(root_dir, 'experiments', 'human_eval', args.output_name)

    log_file = os.path.join(args.output_dir, 'gen.log')
    set_logging(args, log_file)
    set_devices(args)
    set_seed(args)

    tokenizer, model, input_device = load_model(
        'lm' if args.model_type in ('lm', 'text-prompt') else 'prefix',
        args.model_dir, False, args
    )
    model.eval()

    yaml_files = sorted([
        f for f in os.listdir(args.data_dir) if f.endswith('.yaml')
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    for yaml_file in yaml_files:
        yaml_path = os.path.join(args.data_dir, yaml_file)
        output_yaml_path = os.path.join(args.output_dir, yaml_file)
        if os.path.exists(output_yaml_path) and os.path.getsize(output_yaml_path) > 0:
            args.logger.info('Skipping %s (already exists)', yaml_file)
            continue

        with open(yaml_path) as f:
            problem = Problem.load(f)

        prompt = problem.prompt
        stop_tokens = problem.stop_tokens

        args.logger.info('Processing %s', problem.name)

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(input_device)

        gen_kwargs = {
            'do_sample': True,
            'num_return_sequences': args.num_gen,
            'temperature': args.temp,
            'max_new_tokens': args.max_gen_len,
            'top_p': args.top_p,
            'pad_token_id': tokenizer.pad_token_id,
            'use_cache': True,
        }

        if args.model_type == 'prefix':
            control_id = 0 if args.control == 'sec' else 1 if args.control == 'vul' else 0
            gen_kwargs['control_id'] = control_id

        gen_output = model.generate(input_ids, **gen_kwargs)

        input_ids_len = input_ids.shape[1]
        tokens = gen_output[:, input_ids_len:, ...]
        completions_raw = tokenizer.batch_decode(tokens)

        completions = []
        for completion in completions_raw:
            if tokenizer.eos_token in completion:
                completion = completion[:completion.find(tokenizer.eos_token)]
            for stop_tok in stop_tokens:
                stop_idx = completion.find(stop_tok)
                if stop_idx != -1:
                    completion = completion[:stop_idx]
            completions.append(completion)

        problem.completions = completions

        output_yaml_path = os.path.join(args.output_dir, yaml_file)
        with open(output_yaml_path, 'w', encoding='utf-8') as f:
            f.write(Problem.dump(problem))

        args.logger.info('Saved %d completions to %s', len(completions), output_yaml_path)

    args.logger.info('HumanEval generation completed.')


if __name__ == '__main__':
    main()
