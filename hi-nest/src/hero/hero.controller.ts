import { Controller } from '@nestjs/common'

import { HeroService } from './hero.service'

@Controller('heroes')
export class HeroController {
  constructor(public service: HeroService) {}
}
