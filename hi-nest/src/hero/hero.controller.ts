import { Controller } from '@nestjs/common'
import { Crud } from '@nestjsx/crud'

import { Hero } from './hero.entity'
import { HeroService } from './hero.service'

@Crud(Hero)
@Controller('heroes')
export class HeroController {
  constructor(public service: HeroService) {}
}
