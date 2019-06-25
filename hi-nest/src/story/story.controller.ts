import { Controller, Get } from '@nestjs/common'

import { prisma } from '../generated/prisma-client'

@Controller('stories')
export class StoryController {
  constructor(private readonly storyService: StoryService) {}

  @Get()
  async getStories(): Promise<Story[]> {
    return await this.storyService.find()
  }
}
