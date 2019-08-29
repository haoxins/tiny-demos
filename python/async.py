import asyncio

async def main():
  print('hello')
  await asyncio.sleep(10)
  print('world')

asyncio.run(main())
