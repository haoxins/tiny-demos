import { Module } from '@nestjs/common'

import { Topic } from './topic.entity'
import { TopicService } from './topic.service'
import { TopicController } from './topic.controller'

@Module({
  providers: [TopicService],
  controllers: [TopicController],
})
export class TopicModule {}
