import argparse
import asyncio
import sys

print(f'Receive arguments: {sys.argv}')

parser = argparse.ArgumentParser(description='Let us go go go.')
parser.add_argument('--ns', help='The namespace')
parser.add_argument('--name', help='The name')
parser.add_argument('--version', help='The version')

args = parser.parse_args()
print(args)
print(f'{args.ns} {args.name} {args.version}')

async def main():
    print('Hello ...')
    await asyncio.sleep(1)
    print('... World!')

# Python 3.7+
asyncio.run(main())
