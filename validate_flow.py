import re
import json
import jsonschema

##### 1차 CodeBlockJsonParser 검증
class CodeBlockJsonParser():
    """Parse JSON object contained in ```json\n \n```"""
    def loads(self, s: str) -> Any:
        re_res = re.search(JSON_REGEX, s, re.DOTALL)
        if re_res is not None:
            return super().loads(re_res.group(1))
        else:
            return super().loads(s)

##### 2차 BasicParser 검증
def validate_loads(s: str, schema:dict) -> Any:
    """Validate the given string against the schema, return loaded object if valid."""
    obj = json.loads(s)

    try:
        jsonschema.validate(obj,schema)
    except jsonschema.ValidationError as e:
        # for bool, number, integer, None, we try to automatic convert them.
        obj = travese_and_convert(obj,schema)  # get json object recursively with automatic type conversion
        jsonschema.validate(obj,schema)

    return obj

##### 3차 Model Prediction과 Prompt에 주어진 schema로 검증
## CUSTOM class
jsonschema.validate(pred, custom_object['verify_schema'])

## ESCAPE class
jsonschema.validate(pred, custom_object['verify_schema'])


##### 실제 사용 예시 (schemabench/bench/base.py), 이 함수에서 위 코드들을 사용
async def validate(self, pred: str):
    # first load and validate the pred
    try:
        codeblockjsonparser = CodeBlockJsonParser()
        loaded = codeblockjsonparser.loads(pred)
    except Exception as e:
        raise ParserError("Failed to load the pred. "+str(e))
    try:
        pred = validate_loads(
            pred,
            schema=self.question.validate_schema
        )
    except Exception as e:
        raise ValidationError(
            "Failed to validate the pred aginst the schema. " + str(e))

    # validate answer
    judge = self.question.validator(pred, ans if self.question.answer is not None else None, self.question.custom_object)
    if asyncio.coroutines.iscoroutine(judge):
        judge = await judge

    return judge