SEC_LABEL = 'sec'
VUL_LABEL = 'vul'

CWES_TRAINED = [
    'cwe-089',
    'cwe-125',
    'cwe-078',
    'cwe-476',
    'cwe-416',
    'cwe-022',
    'cwe-787',
    'cwe-079',
    'cwe-190',
]

# CWE name -> control_id (sec=0, each CWE gets its own index)
CWE_TO_ID = {cwe: i+1 for i, cwe in enumerate(CWES_TRAINED)}
# {'cwe-089':1, 'cwe-125':2, 'cwe-078':3, 'cwe-476':4, 'cwe-416':5,
#  'cwe-022':6, 'cwe-787':7, 'cwe-079':8, 'cwe-190':9}
N_CONTROL = len(CWES_TRAINED) + 1  # 10

CWES_TRAINED_SUBSET = [
    'cwe-089',
    'cwe-125',
    'cwe-078',
    'cwe-022',
    'cwe-787',
    'cwe-079'
]

CWES_PROMPTS = ['cwe-089']

CWES_GEN_1 = [
    'cwe-119',
    'cwe-502',
    'cwe-732',
    'cwe-798'
]

CWES_GEN_2 = [
    'cwe-020',
    'cwe-094',
    'cwe-116',
    'cwe-117',
    'cwe-209',
    'cwe-215',
    'cwe-312',
    'cwe-327',
    'cwe-377',
    'cwe-611',
    'cwe-643',
    'cwe-777',
    'cwe-918',
]

CWES_DICT = {
    'trained': CWES_TRAINED,
    'trained_subset': CWES_TRAINED_SUBSET,
    'prompts': CWES_PROMPTS,
    'gen_1': CWES_GEN_1,
    'gen_2': CWES_GEN_2,
}

VAL_SCENARIOS = {
    ('cwe-078', '2-py'),
    ('cwe-089', '2-py'),
    ('cwe-125', '2-c'),
    ('cwe-190', '2-c'),
    ('cwe-022', '2-py'),
    ('cwe-787', '2-c'),
}

PROMPTS = [
    'The following code is secure\n',
    'The following code is vulnerable\n',
]

MODEL_DIRS = {
    '350m': 'Salesforce/codegen-350M-multi',
    '2b': 'Salesforce/codegen-2B-multi',
    '6b': 'Salesforce/codegen-6B-multi',
    'incoder': 'facebook/incoder-6B',
    'santa': 'bigcode/santacoder',
}