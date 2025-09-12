from .base import BaseSyntaxBench, Question, Generator, Any, BenchItem
import dataclasses
from SyntaxParser.syntaxes.nl import ARCNLParser
import datasets
import os
import json
from concurrent.futures import ThreadPoolExecutor

@dataclasses.dataclass
class ARCQuestion(Question):
    def load_answer(self):
        return {
            "thought": '',
            "answer": self.answer
        }

class ARCTester(BaseSyntaxBench):
    def __init__(self, n_shots: int = 3, subset: bool = True):
        super().__init__(ARCNLParser("NL"),n_shots)
        
        self.dataset = datasets.load_dataset(
            "allenai/ai2_arc",
            "ARC-Challenge",
            split=os.getenv("BENCHMARK_SPLIT", "test"),
            cache_dir=".cache",
        )
        
        self.subset = subset
        if subset:
            self.dataset.shuffle(seed = int(os.environ.get("BENCHMARK_SUBSET_SEED", 42)))
            self.dataset = self.dataset.select(range(len(self.parsers)*100))

        with open(os.path.join(os.path.dirname(__file__), 'arc_shots_gpt.json'), 'r') as f:
            self.fewshots = json.load(f)

        self.schema = {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "put your thought here"
                },
                "answer": {
                	"type": "string",
                    "description": "put your answer here, Options only, e.g. A",
                    "enum": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                }
            },
            "required": ["thought", "answer"],
        }

    def validate(self, pred, ans, custom_object):
        return pred["answer"] == ans["answer"]

    def __getitem__(self,idx):
        q = ARCQuestion(
            dataset="ARC-CHALLENGE*" if self.subset else "ARC-CHALLENGE",
            query=self.dataset[idx]["question"] + "\n" +
                '\n'.join([
                    f"{l}: {t}"
                    for l, t in zip(self.dataset[idx]["choices"]["label"], self.dataset[idx]["choices"]["text"])
                ]),
            answer=self.dataset[idx]["answerKey"],
            answerparser=self.nlparser,
            validate_schema=self.schema,
            validator=self.validate
        )
        return q
    
    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Generator[BenchItem, Any, None]:
        syntaxes = list(self.parsers.keys())

        examples = []
        for item in self.fewshots:
            examples.append(
                Question(
                    dataset="ARC-CHALLENGE*" if self.subset else "ARC-CHALLENGE",
                    query=item["question"] + "\n" +
                        '\n'.join([
                            f"{l}: {t}"
                            for l, t in item["choices"].items()
                        ]),
                    answer=f"{item['thought']}\nSo the answer is: {item['answer']}.",
                    answerparser=self.nlparser,
                    validate_schema=self.schema,
                    validator=self.validate
                )
            )

        for idx in range(len(self)):
            selected_syntax = syntaxes[idx % len(syntaxes)]
            q = self[idx]
            yield BenchItem(
                question=q,
                examples=examples,
                syntax=selected_syntax,
                syntaxparser=self.parsers[selected_syntax]
            )
    