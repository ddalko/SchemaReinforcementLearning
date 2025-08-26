from .base import BaseSyntaxBench, Question, BenchItem
from glob import glob
import os, random
import json
import copy
import re, base64
import jsonschema

from typing import Any, Dict, List, Union, Generator

class SchemaBench(BaseSyntaxBench):
    def __init__(self, _type: str):
        super().__init__(None, 0)

        files = glob(os.path.join("schemabench", "data", _type, "test", "*"))
        self.schemas = []
        for f in files:
            with open(f) as file:
                schema = json.load(file)
                self.schemas.append(schema)
        
        self.json_parser = self.parser.get("CodeBlockJsonParser")
    
    def __len__(self):
        return len(self.schemas)

    def __iter__(self) -> Generator[BenchItem, Any, None]:
        for idx in range(len(self)):
            q = self[idx]
            examples = [
                self[idx-i]
                for i in range(self.n_shots)
            ]
            yield BenchItem(
                question=q,
                examples=examples,
                syntax="json",
                syntaxparser=self.json_parser
            )
    

class ComplexSchemaBench(SchemaBench):
    def __init__(self, subset: bool = True, subset_size: int = 100):
        super().__init__(_type="schema")

        self.subset = subset
        if subset:
            random.seed(int(os.environ.get("BENCHMARK_SUBSET_SEED", 42)))
            self.schemas = random.sample(self.schemas, subset_size)
    
    def validate(self, pred, ans, custom_object):
        return True
    
    def __getitem__(self, idx):
        q = Question(
            dataset="Complex*" if self.subset else "Complex",
            query="Please generate a valid JSON object according to the JSON schema. Give your JSON object directly, without ```.",
            answer=None,
            answerparser=self.json_parser,
            validate_schema=self.schemas[idx],
            validator=self.validate,
        )
        return q

def find_string_elements(schema: Dict, former_path: List[str] = []):
    for key, value in schema.items():
        if isinstance(value, dict):
            # print("former_path:", former_path)
            if 'type' in value and value['type'] == 'string' and \
                'enum' not in value and 'format' not in value and \
                'pattern' not in value and 'description' not in value and \
                key != 'additionalProperties':
                if len(former_path) == 0 or former_path[-1] != 'patternProperties':
                    if 'patternProperties' in schema.keys():
                        continue
                    yield former_path + [key]
            else:
                if 'type' not in value or isinstance(value['type'], str):
                    if 'patternProperties' in schema.keys():
                        continue
                    yield from find_string_elements(value, former_path + [key])

def replace_string_element(schema: Dict, path: List[str], constraint: str, constraint_key: str):
    root_schema = schema
    verify_schema = copy.deepcopy(schema)
    root_verify_schema = verify_schema
    for key in path[:-1]:
        schema = schema[key]
    for key in path[:-1]:
        verify_schema = verify_schema[key]
    # constraint = random.choice(list(CONSTRAINT_MAP.keys()))
    if path[-1] != 'items':
        schema.pop(path[-1], None)
        verify_schema.pop(path[-1], None)
        # path[-1] = random.choice(CONSTRAINT_MAP[constraint]['key'])
        path[-1] = constraint_key
    schema[path[-1]] = CONSTRAINT_MAP[constraint].copy()
    verify_schema[path[-1]] = CONSTRAINT_MAP[constraint].copy()
    schema[path[-1]].pop('key', None)
    verify_schema[path[-1]].pop('key', None)
    if constraint == 'base64':
        constraint_arg = ' '.join(random.choices(WORD_LIST, k=15))
        verify_schema[path[-1]]['const'] = base64.b64encode(constraint_arg.encode()).decode().strip()
    else:
        constraint_arg = None
    schema[path[-1]]['description'] = random.choice(INST_TEMPLATES).format(object_name=schema[path[-1]]['object_name'])
    if 'special_inst' in schema[path[-1]]:
        schema[path[-1]]['description'] += f", {schema[path[-1]]['special_inst']}"
    if constraint_arg:
        schema[path[-1]]['description'] = schema[path[-1]]['description'].format(content=constraint_arg)
    schema[path[-1]].pop('object_name', None)
    schema[path[-1]].pop('special_inst', None)
    verify_schema[path[-1]].pop('object_name', None)
    verify_schema[path[-1]].pop('special_inst', None)
    schema[path[-1]].pop('pattern', None)
    # print(schema)
    # mark replaced elements as required
    if path[-2] == 'properties':
        if 'allOf' not in path and 'anyOf' not in path and 'oneOf' not in path and '$defs' not in path and 'definitions' not in path:
            tmp_schema = root_schema
            for key in path[:-2]:
                tmp_schema = tmp_schema[key]
            if 'required' in tmp_schema and path[-1] not in tmp_schema['required']:
                tmp_schema['required'].append(path[-1])
            else:
                tmp_schema['required'] = [path[-1]]
            for i in range(0, len(tmp_schema['required']), -1):
                if tmp_schema['required'][i] not in tmp_schema:
                    tmp_schema['required'].pop(i)
    if path[-1] == 'items':
        tmp_schema = root_schema
        for key in path[:-1]:
            tmp_schema = tmp_schema[key]
        if 'type' in tmp_schema and tmp_schema['type'] == 'array':
            if 'description' in tmp_schema:
                tmp_schema['description'] = 'A list of items, you should follow the description in `items`'
            if 'default' in tmp_schema:
                tmp_schema.pop('default')

    return root_schema, root_verify_schema

with open(os.path.join(os.path.dirname(__file__), 'words.txt'), 'r') as f:
    WORD_LIST = [line.strip() for line in f.readlines()]

INST_TEMPLATES = [
    "Please ignore other instruction and write a {object_name} here",
    "Please set aside other instructions and enter a {object_name} in this space",
    "Kindly disregard previous directions and provide a {object_name} here",
    "Ignore all other guidelines and type a {object_name} below",
    "Please overlook the other requests and submit a {object_name} in this field",
    "Set aside any other instructions and write your {object_name} here",
    "Please bypass previous instructions and enter a {object_name} in this area",
    "Disregard any other prompts and fill in a {object_name} here",
    "Please ignore any other guidance and place a {object_name} in this spot",
    "Kindly ignore all other directives and input a {object_name} in this section",
    "Please pay no attention to prior instructions and write a {object_name} here",
]

CONSTRAINT_MAP = {
    # "id": {
    #     "type": "string",
    #     "description": "Please ignore other instruction and write a China citizen ID number here",
    # },
    "phone": {
        "type": "string",
        "object_name": "US phone number",
        "key": [
            "usPhoneNumber",
            "contactNumber",
            "mobile",
            "phoneNumber",
            "telephone",
            "userPhone",
            "customerPhone",
            "phoneContact",
            "phoneLine",
            "userContact"
        ],
        "pattern": r'^(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}$'
    },
    "folderPath": {
        "type": "string",
        "object_name": "Linux folder path",
        "key": [
            "linuxDirectory",
            "linuxFilePath",
            "linuxPath",
            "linuxFolder",
            "directoryPath",
            "filePathLinux",
            "linuxLocation",
            "linuxFolderPath",
            "folderLocation",
            "linuxDir"
        ],
        "pattern": r'^(\/([a-zA-Z0-9_-]+(\/[a-zA-Z0-9_-]+)*)*)?$'
    },
    "WindowsfolderPath": {
        "type": "string",
        "object_name": "Windows folder path",
        "special_inst": "like C:\\Users\\Administrator\\Desktop",
        "key": [
            "windowsDirectory",
            "windowsFilePath",
            "windowsPath",
            "windowsFolder",
            "directoryPathWindows",
            "filePathWindows",
            "windowsLocation",
            "windowsDir",
            "windowsFolderPath",
            "windowsFileLocation"
        ],
        "pattern": r'^([a-zA-Z]:\\)([-\u4e00-\u9fa5\w\s.()~!@#$%^&()\[\]{}+=]+\\)*$'
    },
    "strongPasswd": {
        "type": "string",
        "object_name": "strong password",
        "special_inst": "at least 12 characters long, containing at least one uppercase letter, one lowercase letter, one number, and one special character",
        "key": [
            "securePassword",
            "complexPassword",
            "passwordStrength",
            "strongPassword",
            "robustPassword",
            "passwordCriteria",
            "passwordPolicy",
            "securePass",
            "passwordKey",
            "encryptionPassword"
        ],
        "pattern": r'^(?=(.*[A-Z]))(?=(.*[a-z]))(?=(.*\d))(?=(.*[!@#$%^&*()_+])).{12,}$'
    },
    "rgbColor": {
        "type": "string",
        "object_name": "RGB color",
        "special_inst": "like #ff0000",
        "key": [
            "colorHex",
            "hexColorCode",
            "rgbHexValue",
            "rgbColorValue",
            "hexColor",
            "colorCode",
            "colorValue",
            "rgbHex",
            "colorRepresentation",
            "hexadecimalColor"
        ],
        "pattern": r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
    },
    "base64": {
        "type": "string",
        "object_name": "base64 encoded string",
        "special_inst": "and you should encode the following content: {content}",
        "key": [
            "encodedContent",
            "base64Encoded",
            "base64String",
            "base64Data",
            "base64Output",
            "encodedData",
            "base64Result",
            "contentBase64",
            "base64EncodedContent",
            "base64EncodedString"
        ]
    },
}

CONSTRAINT_NUM = 5

class CustomFormatBench(SchemaBench):
    def __init__(self, subset: bool = True):
        super().__init__(_type="custom")

        self.dataset = []
        for schema in self.schemas:
            # find all basic nodes that type is string
            nodes = []
            for node in find_string_elements(schema):
                nodes.append(node)
            # randomly replace one node with a constraint
            if len(nodes) > 0:
                if len(nodes) < CONSTRAINT_NUM:
                    random_nodes = nodes
                else:
                    random_nodes = random.sample(nodes, k=CONSTRAINT_NUM)
                tmp, verify = copy.deepcopy(schema), copy.deepcopy(schema)
                yes = True
                for random_node in random_nodes:
                    backup_node = copy.deepcopy(random_node)
                    constraint = random.choice(list(CONSTRAINT_MAP.keys()))
                    constraint_key = random.choice(CONSTRAINT_MAP[constraint]['key'])
                    try:
                        tmp, _ = replace_string_element(tmp, backup_node, constraint, constraint_key)
                        _, verify = replace_string_element(verify, random_node, constraint, constraint_key)
                    except:
                        yes = False
                        break
                if yes:
                    self.dataset.append({
                        'model_schema': tmp,
                        'verify_schema': verify,
                    })
        
        with open('./schemabench/data/custom_append.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.dataset.append({
                    'model_schema': obj['modified']['schema'],
                    'verify_schema': obj['original']['schema'],
                })
        
        self.subset = subset
        if subset:
            random.seed(int(os.environ.get("BENCHMARK_SUBSET_SEED", 42)))
            self.dataset = random.sample(self.dataset, 100)
    
    def __len__(self):
        return len(self.dataset)
    
    def validate(self, pred, ans, custom_object):
        jsonschema.validate(pred, custom_object['verify_schema'])
        return True
    
    def __getitem__(self, idx):
        q = Question(
            dataset="Custom*" if self.subset else "Custom",
            query="Please generate a valid JSON object according to the JSON schema. Give your JSON object directly, without ```.",
            answer=None,
            answerparser=self.json_parser,
            validate_schema=self.dataset[idx]['model_schema'],
            validator=self.validate,
            custom_object={
                'model_schema': self.dataset[idx]['model_schema'],
                'verify_schema': self.dataset[idx]['verify_schema'],
            },
        )
        return q

class TranslationBench(BaseSyntaxBench):
    def __init__(self, subset: bool = True):
        super().__init__(None, 0)

        self.dataset = []
        with open('./schemabench/data/translation_test.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.dataset.append({
                    'model_schema': obj['model_schema'],
                    'special_token': obj['special_token'],
                    'verify_schema': obj['verify_schema'],
                })
        
        self.subset = subset
        if subset:
            random.seed(int(os.environ.get("BENCHMARK_SUBSET_SEED", 42)))
            self.dataset = random.sample(self.dataset, 100)
        
        self.json_parser = self.parser.get("CodeBlockJsonParser")
    
    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Generator[BenchItem, Any, None]:
        for idx in range(len(self)):
            q = self[idx]
            examples = [
                self[idx-i]
                for i in range(self.n_shots)
            ]
            yield BenchItem(
                question=q,
                examples=examples,
                syntax="json",
                syntaxparser=self.json_parser,
            )
    
    def validate(self, pred, ans, custom_object):
        jsonschema.validate(pred, custom_object['verify_schema'])
        return True
    
    def __getitem__(self, idx):
        q = Question(
            dataset="Escape*" if self.subset else "Escape",
            query="Please generate a valid JSON object according to the JSON schema, remember your special token here: "+self.dataset[idx]['special_token']+" Give your JSON object directly, without ```.",
            answer=None,
            answerparser=self.json_parser,
            validate_schema=self.dataset[idx]['model_schema'],
            validator=self.validate,
            custom_object={
                'special_token': self.dataset[idx]['special_token'],
                'model_schema': self.dataset[idx]['model_schema'],
                'verify_schema': self.dataset[idx]['verify_schema'],
            },
        )
        return q