import { Module } from '@nestjs/common'

import { StoryController } from './story.controller'
import { StoryService } from './story.service'
import { Story } from './story.entity'

@Module({
  controllers: [StoryController],
  providers: [StoryService],
})
export class StoryModule {}
