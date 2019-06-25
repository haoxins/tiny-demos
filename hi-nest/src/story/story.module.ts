import { Module } from '@nestjs/common'

import { StoryController } from './story.controller'

@Module({
  controllers: [StoryController],
})
export class StoryModule {}
