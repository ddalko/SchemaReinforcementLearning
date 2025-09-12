import dataclasses
import os
import json
import jsonlines
import tqdm
import asyncio
import SyntaxParser
from typing import Any, Callable, Coroutine, Literal, Generator
import datasets

class AnswerInvalid(Exception):
    pass

class ParserError(Exception):
    pass

class ValidationError(Exception):
    pass

@dataclasses.dataclass
class Question:
    dataset: str
    query: str
    answer: str
    answerparser: SyntaxParser.BasicParser
    validate_schema: dict
    validator: Callable[[str|dict, str|dict], bool | Coroutine[Any, Any, bool]]
    custom_object: Any = None

    def load_answer(self):
        return self.answerparser.validate_loads(self.answer, self.validate_schema)
    
@dataclasses.dataclass
class BenchItem:
    question: Question
    examples: list[Question]
    syntax: str
    syntaxparser: SyntaxParser.BasicParser

    def get_prompt(self) -> list[dict]:
        sys_msg = [
            f"You should generate answer with given {self.syntax} format.",
            self.syntaxparser.intro,
        ]

        try:
            if len(self.examples) > 0:
                example_ans = self.examples[0].load_answer()
        except Exception as e:
            raise AnswerInvalid(
                f"The answer of the query\n{self.examples[0].query}\n is invalid, please check the answerparser or dataset. " + str(e))

        if len(self.examples) > 0:
            sys_msg.extend([
                f"<Example> Here is a example of what generated content should looks like with {self.syntax} format:",
                self.syntaxparser.validate_dumps(
                    example_ans, self.question.validate_schema),
                "</Example>",
            ])
        sys_msg.extend([
            "<Schema> Here are the json-schema of the content format:",
            json.dumps(self.question.validate_schema),
            "</Schema>",
        ])


        sys_msg = '\n'.join(sys_msg)

        msgs = [
            {"role": "system", "content": sys_msg}
        ]
        for example in self.examples[1:]:
            msgs.append({"role": "user", "content": example.query})
            msgs.append({
                "role": "assistant",
                "content": self.syntaxparser.validate_dumps(
                    example.load_answer(),
                    self.question.validate_schema)
            })
        msgs.append({"role": "user", "content": self.question.query})

        return msgs

    async def validate(self, pred: str):
        try:
            if self.question.answer is not None:
                ans = self.question.load_answer()
        except Exception as e:
            raise AnswerInvalid(
                "The answer of the query is invalid, please check the answerparser or dataset. " + str(e))

        # first load and validate the pred
        try:
            loaded = self.syntaxparser.loads(pred)
        except Exception as e:
            raise ParserError("Failed to load the pred. "+str(e))
        try:
            pred = self.syntaxparser.validate_loads(
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


class BaseSyntaxBench:
    def __init__(self, nlparser: SyntaxParser.BasicParser, n_shots: int = 1,):
        
        self.nlparser = nlparser
        self.dataset: datasets.Dataset = None
        self.n_shots = n_shots
        self.parser = SyntaxParser.SyntaxesParser(init_all_parser=True)
        
        if os.getenv("BENCHNL","False") == "True":
            self.parsers = {
                "nl": self.nlparser
            }
        else:
            
            self.parsers = {
                "json": self.parser.get("CodeBlockJsonParser"),
                # "yaml": self.parser.get("SimplifiedYamlParser"),
                # "toml": self.parser.get("StandardTomlParser"),
                # "xml": self.parser.get("StandardXMLParser"),
                # "hcl": self.parser.get("StandardHCLParser"),
            }
        self.statics = {}

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Generator[BenchItem, Any, None]:
        syntaxes = list(self.parsers.keys())
        for idx in range(len(self)):
            selected_syntax = syntaxes[idx % len(syntaxes)]
            q = self[idx]
            examples = [
                self[idx-i]
                for i in range(self.n_shots)
            ]
            yield BenchItem(
                question=q,
                examples=examples,
                syntax=selected_syntax,
                syntaxparser=self.parsers[selected_syntax]
            )

    async def run(self,
            save_path:str,
            model_completion_func: Callable[[list[dict]], Coroutine[Any, Any, str]]
        ):
        out = jsonlines.Writer(open(save_path, "w", encoding="utf-8"))
        pbar = tqdm.tqdm(total=len(self), ncols=150)

        scores = []

        async def forward(benchitem:BenchItem,):
            res = None
            score = False
            try:
                res = await model_completion_func(benchitem.get_prompt())
                score = await asyncio.wait_for(benchitem.validate(res),timeout=10)
            except Exception as e:
                scores.append(False)
                error_type = e.__class__.__name__
                self.statics[error_type] = self.statics.get(error_type, 0) + 1
                out.write({
                    "error": {
                        "type": error_type,
                        "message": str(e),
                    },
                    "dataset": benchitem.question.dataset,
                    "question": benchitem.question.query,
                    "gt": benchitem.question.answer,
                    "ans": res,
                    "syntax": benchitem.syntax,
                    "score": False,
                    "schema": benchitem.question.validate_schema,
                })

                pbar.set_description("Accuracy: {:.6f}".format(sum(scores)/len(scores)))
                pbar.update(1)
                return
            
            scores.append(score)
            out.write({
                "dataset": benchitem.question.dataset,
                "question": benchitem.question.query,
                "gt": benchitem.question.answer,
                "ans": res,
                "syntax": benchitem.syntax,
                "score": score,
                "schema": benchitem.question.validate_schema,
            })
            pbar.set_description("Accuracy: {:.6f}".format(sum(scores)/len(scores))
            )
            pbar.update(1)

            if (pbar.n+1) % 100 == 0:
                print(f"Error stats: {self.statics}")

        tasks = []
        for benchitem in self:
            tasks.append(asyncio.create_task(forward(benchitem)))

        for t in asyncio.as_completed(tasks):
            try:
                await t
            except:
                import traceback
                traceback.print_exc()
            