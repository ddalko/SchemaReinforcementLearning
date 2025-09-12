from .base import BaseSyntaxBench, Question
from SyntaxParser.syntaxes.nl import GSMNLParser
import datasets
import os


class GSM8KSyntaxBench(BaseSyntaxBench):
    def __init__(self, n_shots: int = 3, subset: bool = True):
        super().__init__( GSMNLParser("NL"),n_shots)

        self.dataset = datasets.load_dataset(
            "openai/gsm8k",
            "main",
            split="test",
            cache_dir=".cache",
        )
        self.subset = subset
        if subset:
            self.dataset.shuffle(seed = int(os.environ.get("BENCHMARK_SUBSET_SEED", 42)))
            self.dataset = self.dataset.select(range(len(self.parsers)*100))
        self.schema = {
            "type": "object",
            "properties":{
                "thought": {
                    "type": "string",
                    "description": "put your thought here"
                },
                "answer": {
                	"type": "number",
                    "description": "put your answer here, integer only"
                }
            },
            "required": ["thought", "answer"],
        }
    
    def validate(self, pred, ans, custom_object):
        return pred["answer"] == ans["answer"]
    
    def __getitem__(self,idx):
        q = Question(
            dataset="GSM8K*" if self.subset else "GSM8K",
            query=self.dataset[idx]["question"],
            answer=self.dataset[idx]["answer"],
            answerparser=self.nlparser,
            validate_schema=self.schema,
            validator=self.validate
        )
        return q
    
    def __len__(self):
        return len(self.dataset)
