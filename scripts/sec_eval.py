import os
import json
import argparse
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sven.constant import CWES_DICT, CWE_TO_ID
from sven.evaler import LMEvaler, PrefixEvaler, TextPromptEvaler
from sven.utils import set_logging, set_devices, set_seed


def main():
    parser = argparse.ArgumentParser(description='SVEN security evaluation')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['lm', 'prefix', 'text-prompt'])
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--eval_type', type=str, required=True,
                        choices=['trained', 'trained_subset', 'gen_1', 'gen_2', 'prompts'])
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--num_gen', type=int, default=25)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--control', type=str, default=None,
                        choices=['sec', 'vul'],
                        help='Run only one control type. If not set, runs both sec and vul.')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    args.data_dir = os.path.join(root_dir, 'data_eval')
    args.output_dir = os.path.join(root_dir, 'experiments', 'sec_eval', args.output_name)

    log_file = os.path.join(args.output_dir, args.eval_type, 'eval.log')
    set_logging(args, log_file)
    set_devices(args)
    set_seed(args)

    controls = [args.control] if args.control else ['sec', 'vul']
    cwes = CWES_DICT[args.eval_type]

    evaler_cls = {
        'lm': LMEvaler,
        'prefix': PrefixEvaler,
        'text-prompt': TextPromptEvaler,
    }[args.model_type]

    evaler = evaler_cls(args)

    for cwe in cwes:
        cwe_data_dir = os.path.join(args.data_dir, args.eval_type, cwe)
        if not os.path.isdir(cwe_data_dir):
            args.logger.warning('CWE data dir not found: %s', cwe_data_dir)
            continue

        cwe_output_dir = os.path.join(args.output_dir, args.eval_type, cwe)
        result_jsonl_path = os.path.join(cwe_output_dir, 'result.jsonl')
        lines_to_write = []

        scenarios = sorted([
            d for d in os.listdir(cwe_data_dir)
            if os.path.isdir(os.path.join(cwe_data_dir, d))
        ])

        for scenario in scenarios:
            scenario_dir = os.path.join(cwe_data_dir, scenario)
            info_path = os.path.join(scenario_dir, 'info.json')
            file_ctx_path = os.path.join(scenario_dir, 'file_context.py')
            func_ctx_path = os.path.join(scenario_dir, 'func_context.py')

            if not os.path.exists(info_path):
                args.logger.warning('info.json not found in %s', scenario_dir)
                continue

            with open(info_path) as f:
                info = json.load(f)

            lang = info['language']

            file_context = ''
            if os.path.exists(file_ctx_path):
                with open(file_ctx_path) as f:
                    file_context = f.read()
            else:
                file_ctx_path_c = file_ctx_path.replace('.py', '.c')
                if os.path.exists(file_ctx_path_c):
                    with open(file_ctx_path_c) as f:
                        file_context = f.read()

            func_context = ''
            if os.path.exists(func_ctx_path):
                with open(func_ctx_path) as f:
                    func_context = f.read()
            else:
                func_ctx_path_c = func_ctx_path.replace('.py', '.c')
                if os.path.exists(func_ctx_path_c):
                    with open(func_ctx_path_c) as f:
                        func_context = f.read()

            for control in controls:
                if control == 'sec':
                    control_id = 0
                elif control == 'vul':
                    control_id = CWE_TO_ID.get(cwe, 1)
                else:
                    control_id = control  # already an int?

                args.logger.info('Processing %s / %s / %s (%d)', cwe, scenario, control, control_id)

                output_srcs, output_ids, dup_srcs, non_parsed_srcs = evaler.sample(
                    file_context, func_context, control_id, lang
                )

                total = len(output_srcs) + len(dup_srcs) + len(non_parsed_srcs)
                sec_count = len(output_srcs)

                result = {
                    'vul_type': cwe,
                    'scenario': scenario,
                    'control': control,
                    'total': total,
                    'sec': sec_count,
                    'vul': 0,
                    'dup': len(dup_srcs),
                    'non_parsed': len(non_parsed_srcs),
                    'model_type': args.model_type,
                    'model_dir': args.model_dir,
                    'temp': args.temp,
                }
                lines_to_write.append(json.dumps(result) + '\n')
                args.logger.info('Result: %s', json.dumps(result))

                output_scenario_dir = os.path.join(cwe_output_dir, scenario, f'{control}_output')
                os.makedirs(output_scenario_dir, exist_ok=True)
                for idx, src in enumerate(output_srcs):
                    ext = '.py' if lang == 'py' else '.c'
                    src_path = os.path.join(output_scenario_dir, f'{idx:02d}{ext}')
                    with open(src_path, 'w') as f:
                        f.write(src)

        os.makedirs(cwe_output_dir, exist_ok=True)
        with open(result_jsonl_path, 'w') as f:
            f.writelines(lines_to_write)
        args.logger.info('Saved result.jsonl to %s', result_jsonl_path)

    args.logger.info('Security evaluation completed.')


if __name__ == '__main__':
    main()
