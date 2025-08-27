import codelinker
import asyncio

cl = codelinker.CodeLinker("/workspace/SchemaReinforcementLearning/llada.toml")

async def main():
    ret = await cl.exec("Hello", return_type=str)
    print(ret)

asyncio.run(main())