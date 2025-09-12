from .base import BaseSyntaxBench, Question
from SyntaxParser.syntaxes.nl import MATHNLParser
import datasets
import os


class MATH500SyntaxBench(BaseSyntaxBench):
    def __init__(self, n_shots: int = 3, subset: bool = True):
        super().__init__(MATHNLParser("NL"), n_shots)

        self.dataset = datasets.load_dataset(
            "HuggingFaceH4/MATH-500",
            split=os.getenv("BENCHMARK_SPLIT", "test"),
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
                	"type": "string",
                    "description": "put your answer here"
                }
            },
            "required": ["thought", "answer"],
        }

    
    def validate(self, pred, ans, custom_object):
        try:
            return eval(pred["answer"]) == eval(ans["answer"])
        except:
            return str(pred["answer"]) == str(ans["answer"])

    
    def __getitem__(self, idx):
        q = Question(
            dataset="MATH500*" if self.subset else "MATH500",
            query=self.dataset[idx]["problem"],
            answer=self.dataset[idx]["solution"],
            answerparser=self.nlparser,
            validate_schema=self.schema,
            validator=self.validate
        )
        return q
    
    def __len__(self):
        return len(self.dataset)