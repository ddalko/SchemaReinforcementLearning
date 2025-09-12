from .base import BaseSyntaxBench, Question, Generator, Any, BenchItem
import dataclasses
from SyntaxParser.syntaxes.nl import MMLUNLParser
import datasets
import os
import json
from concurrent.futures import ThreadPoolExecutor


@dataclasses.dataclass
class MMLUQuestion(Question):
    def load_answer(self):
        return {
            "thought": None,
            "answer": self.answer
        }


class MMLUSyntaxBench(BaseSyntaxBench):
    def __init__(self, n_shots: int = 3, subset: bool = True):
        super().__init__(MMLUNLParser("NL"),n_shots)
        
        with ThreadPoolExecutor(max_workers=32) as executor:
            self.subdatasets = list(executor.map(
            lambda subset: datasets.load_dataset(
                "cais/mmlu",
                subset,
                split=os.getenv("BENCHMARK_SPLIT", "test"),
                cache_dir=".cache",
            ),
            [
                "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
            ]
            ))

        self.subset = subset
        if subset:
            for idx, dataset in enumerate(self.subdatasets):
                dataset.shuffle(
                    seed=int(os.environ.get("BENCHMARK_SUBSET_SEED", 42)))
                self.subdatasets[idx] = dataset.select(
                    range(int(len(self.parsers)*(100/len(self.subdatasets)))))

        with open(os.path.join(os.path.dirname(__file__), 'mmlu_shots_gpt.json'), 'r') as f:
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
                    "enum": ["A", "B", "C", "D"],
                    "description": "put your choice here"
                }
            },
            "required": ["thought", "answer"],
        }

    def validate(self, pred, ans, custom_object):
        return pred["answer"] == ans["answer"]

    def __getitem__(self, idx):
        for subset in self.subdatasets:
            if idx >= len(subset):
                idx -= len(subset)
            else:
                q = MMLUQuestion(
                    dataset="MMLU*" if self.subset else "MMLU",
                    query=subset[idx]["question"] + "\n" +
                    '\n'.join([
                        f"{chr(ord('A')+idx)}: {content}"
                        for idx, content in enumerate(subset[idx]["choices"])
                    ]),
                    answer=chr(subset[idx]["answer"]+ord("A")),
                    answerparser=self.nlparser,
                    validate_schema=self.schema,
                    validator=self.validate
                )
                return q

        raise IndexError(f"Index {idx} out of range.")

    def __len__(self):
        return sum([len(dataset) for dataset in self.subdatasets])

    def __iter__(self) -> Generator[BenchItem, Any, None]:
        syntaxes = list(self.parsers.keys())
        subset_idx = 0
        past_subset_idx = 0
        for idx in range(len(self)):
            if idx >= past_subset_idx + len(self.subdatasets[subset_idx]):
                past_subset_idx += len(self.subdatasets[subset_idx])
                subset_idx += 1

            selected_syntax = syntaxes[idx % len(syntaxes)]
            q = self[idx]

            examples = []
            for item in self.fewshots[subset_idx]:
                examples.append(
                    Question(
                        dataset="MMLU*" if self.subset else "MMLU",
                        query=item["question"] + "\n" + '\n'.join(
                            [f"{chr(ord('A')+idx)}: {content}" for idx, content in enumerate(item["choices"])]),
                        answer=f"{item['thought']}\nSo the answer is: {item['answer']}.",
                        answerparser=self.nlparser,
                        validate_schema=self.schema,
                        validator=self.validate
                    )
                )

            yield BenchItem(
                question=q,
                examples=examples,
                syntax=selected_syntax,
                syntaxparser=self.parsers[selected_syntax]
            )
