import { Controller, Get } from '@nestjs/common'

import { StoryService } from './story.service'
import { Story } from './story.entity'

@Controller('stories')
export class StoryController {
  constructor(private readonly storyService: StoryService) {}

  @Get()
  async getStories(): Promise<Story[]> {
    return await this.storyService.find()
  }
}
