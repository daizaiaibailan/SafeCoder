import os, json

EVAL = r'C:\Users\sven-master\experiments\sec_eval\sec-eval-multi\trained'

for cwe in sorted(os.listdir(EVAL)):
    cd = os.path.join(EVAL, cwe)
    if not os.path.isdir(cd):
        continue
    rj_path = os.path.join(cd, 'result.jsonl')
    if not os.path.exists(rj_path):
        continue

    # Read existing results
    results = []
    with open(rj_path) as f:
        for line in f:
            results.append(json.loads(line))

    # Update with CodeQL data
    for r in results:
        scenario = r['scenario']
        control = r['control']
        sd = os.path.join(cd, scenario)
        if not os.path.isdir(sd):
            continue

        csv_path = os.path.join(sd, f'{control}_codeql.csv')
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                vul_count = sum(1 for _ in f)
        else:
            vul_count = 0

        # sec_output files = total output files
        output_dir = os.path.join(sd, f'{control}_output')
        if os.path.isdir(output_dir):
            total_gen = len([f for f in os.listdir(output_dir) if f.endswith(('.py','.c','.cpp','.cc'))])
        else:
            total_gen = r.get('total', 0)

        r['vul'] = vul_count
        r['sec'] = max(0, total_gen - r.get('dup', 0) - r.get('non_parsed', 0) - vul_count)
        if r['sec'] + r['vul'] + r.get('dup', 0) + r.get('non_parsed', 0) != r['total']:
            r['total'] = r['sec'] + r['vul'] + r.get('dup', 0) + r.get('non_parsed', 0)

    # Write back
    with open(rj_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f'{cwe}: {len(results)} results updated')

print('Done.')
