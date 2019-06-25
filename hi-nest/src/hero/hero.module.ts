import { Module } from '@nestjs/common'

import { HeroController } from './hero.controller'
import { HeroService } from './hero.service'
import { Hero } from './hero.entity'

@Module({
  controllers: [HeroController],
  providers: [HeroService],
})
export class HeroModule {}
