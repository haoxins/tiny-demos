import { Injectable } from '@nestjs/common'
import { Story } from './story.entity'

@Injectable()
export class StoryService {
  constructor(
    @InjectRepository(Story)
    private readonly storyRepository: Repository<Story>,
  ) {}

  find(): Promise<Story[]> {
    return this.storyRepository.find()
  }
}
