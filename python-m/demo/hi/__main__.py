import argparse
import asyncio
import aiohttp
import sys

parser = argparse.ArgumentParser(description='Let us go go go.')
parser.add_argument('--ns', help='The namespace')
parser.add_argument('--name', help='The name')
parser.add_argument('--version', help='The version')

args = parser.parse_args()


async def hello():
  print(f'Receive arguments: {sys.argv}')

  await asyncio.sleep(1)

  print(f'{args.ns} {args.name} {args.version}')

  async with aiohttp.ClientSession() as session:
    async with session.get('https://httpbin.org/json') as response:

      print('Status:', response.status)
      print('Content-type:', response.headers['content-type'])

      text = await response.text()
      print(f'Body: {text}')


loop = asyncio.get_event_loop()
loop.run_until_complete(hello())
