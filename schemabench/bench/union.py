import tqdm.auto
from .base import BaseSyntaxBench, BenchItem, Question
import os

from .gsm8k import GSM8KSyntaxBench
from .math500 import MATH500SyntaxBench
from .mmlu import MMLUSyntaxBench
from .arc import ARCTester
from .schemaBench import ComplexSchemaBench, CustomFormatBench, TranslationBench

import jsonlines, json
import tqdm
import asyncio
from typing import Any, Callable, Coroutine
import random

class UnionSyntaxBench(BaseSyntaxBench):
    def __init__(self, n_shots: int = 3, subset: bool = True, test_category: str = "all", debug: bool = False):
        super().__init__(n_shots)
        allbenches = {
            "MATH500": MATH500SyntaxBench,
            "MMLU": MMLUSyntaxBench,
            "ARC": ARCTester,
            "GSM8K": GSM8KSyntaxBench,
            "COMPLEX": ComplexSchemaBench,
            "CUSTOM": CustomFormatBench, # combined
            "ESCAPE": TranslationBench,
        }
        if debug:
            self.subbenches = {
                "COMPLEX": ComplexSchemaBench(subset=subset, subset_size=5),
            }
            return
        if test_category == "all":
            self.subbenches = allbenches
        elif test_category == "schema":
            self.subbenches = {
                "COMPLEX": ComplexSchemaBench(subset=subset),
                "CUSTOM": CustomFormatBench(subset=subset),
                "ESCAPE": TranslationBench(subset=subset),
            }
        elif test_category == "reasoning":
            self.subbenches = {
                "MATH500": MATH500SyntaxBench(n_shots, subset),
                "MMLU": MMLUSyntaxBench(n_shots, subset),
                "ARC": ARCTester(n_shots, subset),
                "GSM8K": GSM8KSyntaxBench(n_shots, subset),
            }
        elif test_category in ["math", "math500", "mmlu", "arc", "gsm8k"]:
            self.subbenches = {
                test_category.upper(): allbenches[test_category.upper()](n_shots, subset)
            }
        elif test_category in ["complex", "custom", "escape"]:
            self.subbenches = {
                test_category.upper(): allbenches[test_category.upper()](subset)
            }
        else:
            raise ValueError(f"Invalid test category: {test_category}")

    def __len__(self):
        return sum(len(bench) for bench in self.subbenches.values())
    
    def __iter__(self):
        for bench in self.subbenches.values():
            yield from bench
    
    async def run(
        self,
        save_path:str,
        model_completion_func: Callable[[list[dict]], Coroutine[Any, Any, str]]
    ):
        # out = jsonlines.Writer(open(save_path, "w", encoding="utf-8"))

        async def forward(benchitem: BenchItem, pbar: tqdm.tqdm, benchname: str, scores: list):
            res = None
            score = False
            try:
            #     await asyncio.sleep(random.random() * 30)
                # print("start, pbar.n =", pbar.n)
                res = await model_completion_func(benchitem.get_prompt())
                # print("finish...? pbar.n =", pbar.n)
                score = await asyncio.wait_for(benchitem.validate(res), timeout=10)
                # print("finish, pbar.n =", pbar.n)
            except Exception as e:
                scores.append(False)
                error_type = e.__class__.__name__
                self.statics[error_type] = self.statics.get(error_type, 0) + 1
                # print("oops, pbar.n =", pbar.n, error_type)
                pbar.set_description(benchname+" accuracy: {:.6f}".format(sum(scores)/len(scores))+f" {sum(scores)}/{len(scores)}")
                pbar.update(1)
                return {
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
                }
            
            scores.append(score)
            pbar.set_description(benchname+" accuracy: {:.6f}".format(sum(scores)/len(scores))+f" {sum(scores)}/{len(scores)}")
            pbar.update(1)

            # if (pbar.n+1) % 100 == 0:
            #     print(f"Error stats: {self.statics}")

            return {
                "dataset": benchitem.question.dataset,
                "question": benchitem.question.query,
                "gt": benchitem.question.answer,
                "ans": res,
                "syntax": benchitem.syntax,
                "score": score,
                "schema": benchitem.question.validate_schema,
            }

        tasks = []
        for name, bench in self.subbenches.items():
            pbar = tqdm.auto.tqdm(total=len(bench))
            scores = []
            for benchitem in bench:
                tasks.append(asyncio.create_task(forward(benchitem, pbar, name, scores)))

        # results = []
        # for t in asyncio.as_completed(tasks):
        #     try:
        #         res = await t
        #         results.append(res)
        #     except:
        #         import traceback
        #         traceback.print_exc()
        results = await asyncio.gather(*tasks)

        with open(save_path, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res)+'\n')
