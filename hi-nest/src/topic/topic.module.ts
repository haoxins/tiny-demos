import { Module } from '@nestjs/common'
import { TypeOrmModule } from '@nestjs/typeorm'

import { Topic } from './topic.entity'
import { TopicService } from './topic.service'
import { TopicController } from './topic.controller'

@Module({
  imports: [TypeOrmModule.forFeature([Topic])],
  providers: [TopicService],
  controllers: [TopicController],
})
export class TopicModule {}
