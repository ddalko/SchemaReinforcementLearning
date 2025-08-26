import os
import fire
import asyncio
import codelinker
from bench import UnionSyntaxBench

def main(
    save_path: str,
    model: str,
    n: int = 3,
    temperature: float = 0.0,
    seed: int = 42,
    subset: bool = True, 
    test_category: str = "all",
    debug: bool = False,
):
    cl = codelinker.CodeLinker(codelinker.CodeLinkerConfig.from_toml(os.environ.get("CODELINKER_CONFIG", "private.toml")))
    sem = asyncio.Semaphore(int(os.environ.get("CONCURRENCY", 32)))
    # if os.path.exists(save_path):
    #     print(f"File {save_path} already exists. Exiting...")
    #     return
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    async def completion(messages:list[dict]):
        async with sem:
            return await cl.exec(messages=messages, model=model, completions_kwargs={"temperature": temperature, "seed": seed, "max_tokens": 2048})

    bench = UnionSyntaxBench(n_shots=n, subset=subset, test_category=test_category, debug=debug)
    # bench = MATHSyntaxBench(subset=False)
    # print(len(bench))
    # exit()
    asyncio.run(bench.run(save_path, completion))

if __name__ == "__main__":
    fire.Fire(main)