
import { serve } from 'https://deno.land/x/std@v0.2.10/http/server.ts'

const s = serve('0.0.0.0:3000')

async function main() {
  for await (const req of s) {
    req.respond({ body: new TextEncoder().encode('Hello World\n') })
  }
}

main()
