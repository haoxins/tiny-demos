import { Controller, Get } from '@nestjs/common'

import { prisma } from '../generated/prisma-client'

@Controller('stories')
export class StoryController {
  @Get()
  async getStories(): Promise<any[]> {
    return await prisma.stories()
  }
}
