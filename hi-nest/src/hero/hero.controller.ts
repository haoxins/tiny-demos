import { Controller } from '@nestjs/common'

import { Hero } from './hero.entity'
import { HeroService } from './hero.service'

@Controller('heroes')
export class HeroController {
  constructor(public service: HeroService) {}
}
