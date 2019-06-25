import { NestFactory } from '@nestjs/core'
import { AppModule } from './app.module'

import * as helmet from 'helmet'

async function bootstrap() {
  const app = await NestFactory.create(AppModule)

  app.use(helmet())

  console.info('Listening on: 3000')
  await app.listen(3000)
}
bootstrap()
