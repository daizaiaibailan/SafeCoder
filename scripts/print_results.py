import os
import argparse
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sven.metric import SecEval, FuncEval


def main():
    parser = argparse.ArgumentParser(description='SVEN results display')
    parser.add_argument('--eval_type', type=str, required=True,
                        choices=['sec_eval', 'human_eval'])
    parser.add_argument('--eval_dir', type=str, required=True)
    parser.add_argument('--vul_type', type=str, default=None,
                        help='Vulnerability type (for sec_eval). If not set, uses all for the eval_type.')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val'],
                        help='Split for sec_eval (test or val)')
    parser.add_argument('--sec_eval_type', type=str, default='trained',
                        choices=['trained', 'trained_subset', 'gen_1', 'gen_2', 'prompts'],
                        help='Evaluation type for sec_eval')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    if not os.path.isabs(args.eval_dir):
        args.eval_dir = os.path.join(root_dir, 'experiments', args.eval_type, args.eval_dir)

    if args.eval_type == 'sec_eval':
        evaluator = SecEval(
            eval_dir=args.eval_dir,
            eval_type=args.sec_eval_type,
            vul_type=args.vul_type,
            split=args.split,
        )
        evaluator.pretty_print()
    elif args.eval_type == 'human_eval':
        evaluator = FuncEval(eval_dir=args.eval_dir)
        evaluator.pretty_print()
    else:
        raise ValueError(f'Unknown eval_type: {args.eval_type}')


if __name__ == '__main__':
    main()
