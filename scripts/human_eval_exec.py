import os, argparse, sys, time, yaml
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sven.human_eval.problem_yaml import Problem
from sven.human_eval.containerized_eval import eval_string_script
from sven.utils import set_logging

def main():
    parser = argparse.ArgumentParser(description='SVEN HumanEval execution')
    parser.add_argument('--output_name', type=str, required=True)
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    eval_dir = os.path.join(root_dir, 'experiments', 'human_eval', args.output_name)
    log_file = os.path.join(eval_dir, 'exec.log')
    set_logging(args, log_file)

    yaml_files = sorted([f for f in os.listdir(eval_dir) if f.endswith('.yaml') and not f.endswith('.results.yaml')])

    for yaml_file in yaml_files:
        yaml_path = os.path.join(eval_dir, yaml_file)
        results_yaml_path = yaml_path.replace('.yaml', '.results.yaml')
        if os.path.exists(results_yaml_path):
            args.logger.info('Skipping %s', yaml_file)
            continue

        with open(yaml_path) as f:
            problem = Problem.load(f)

        args.logger.info('Executing %s with %d completions', problem.name, len(problem.completions))

        results_list = []
        for completion in problem.completions:
            program = problem.prompt + completion + '\n' + problem.tests
            rd = eval_string_script(problem.language, program)
            results_list.append({
                'program': rd['program'],
                'stdout': rd['stdout'],
                'stderr': rd['stderr'],
                'exit_code': rd['exit_code'],
                'status': rd['status'],
                'timestamp': int(time.time()),
            })

        output = {'name': problem.name, 'language': problem.language, 'results': results_list}
        with open(results_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(output, f, default_flow_style=False, allow_unicode=True)

        n_ok = sum(1 for r in results_list if r['status'] == 'OK')
        args.logger.info('Saved %s: %d/%d passed', problem.name, n_ok, len(results_list))

    args.logger.info('HumanEval execution completed.')

if __name__ == '__main__':
    main()
