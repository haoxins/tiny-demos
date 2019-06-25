import { Injectable } from '@nestjs/common'

import { Hero } from './hero.entity'

@Injectable()
export class HeroService extends RepositoryService<Hero> {
  constructor(@InjectRepository(Hero) repo) {
    super(repo)
  }
}
