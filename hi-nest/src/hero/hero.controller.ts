import { Controller } from '@nestjs/common'

import { prisma } from '../generated/prisma-client'

@Controller('heroes')
export class HeroController {
  constructor(public service: HeroService) {}
}
